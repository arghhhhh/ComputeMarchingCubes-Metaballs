using UnityEngine;
using UnityEngine.Rendering;

namespace MarchingCubes {

//
// Isosurface mesh builder with the marching cubes algorithm
//
// The vertex/index buffers are sized to the triangle budget, but the submesh
// is trimmed to the number of triangles actually produced: the GPU counter is
// copied out after the reconstruction kernel and read back asynchronously
// (one to a few frames stale), then the submesh index count is set to that
// count plus a safety margin, so the renderer only processes the trimmed
// range. The clear kernel zeroes just the slack inside that range, so a
// stale-high count still draws degenerate triangles rather than garbage.
//
sealed class MeshBuilder : System.IDisposable
{
    #region Public members

    public Mesh Mesh => _mesh;

    // Triangles produced by the last completed readback.
    public int LastTriangleCount { get; private set; }

    // Triangles currently exposed through the submesh.
    public int SubmeshTriangleCount => _submeshTriangles;

    public MeshBuilder(int x, int y, int z, int budget, ComputeShader compute)
      => Initialize((x, y, z), budget, compute);

    public MeshBuilder(Vector3Int dims, int budget, ComputeShader compute)
      => Initialize((dims.x, dims.y, dims.z), budget, compute);

    public void Dispose()
      => ReleaseAll();

    public void BuildIsosurface(ComputeBuffer voxels, float target, float scale)
      => RunCompute(voxels, target, scale);

    #endregion

    #region Private members

    static readonly int DimsId = Shader.PropertyToID("Dims");
    static readonly int MaxTriangleId = Shader.PropertyToID("MaxTriangle");
    static readonly int ScaleId = Shader.PropertyToID("Scale");
    static readonly int IsovalueId = Shader.PropertyToID("Isovalue");
    static readonly int TriangleTableId = Shader.PropertyToID("TriangleTable");
    static readonly int VoxelsId = Shader.PropertyToID("Voxels");
    static readonly int VertexBufferId = Shader.PropertyToID("VertexBuffer");
    static readonly int IndexBufferId = Shader.PropertyToID("IndexBuffer");
    static readonly int CounterId = Shader.PropertyToID("Counter");

    // Submesh sizing: last readback * (1 + MarginRatio) + MarginTriangles,
    // grown once the count eats into half of that margin, shrunk only once
    // the wanted size drops below ShrinkRatio of the current range
    // (hysteresis both ways keeps SetSubMesh calls rare while the surface
    // animates).
    //
    // Growing exposes a range the clear kernel has not zeroed at the new
    // size, which can still hold triangles from an older, larger surface.
    // Callers that skip BuildIsosurface while idle should rebuild once when
    // SubmeshTriangleCount grows.
    const float MarginRatio = 0.25f;
    const int MarginTriangles = 512;
    const float ShrinkRatio = 0.6f;

    (int x, int y, int z) _grids;
    int _triangleBudget;
    ComputeShader _compute;
    int _submeshTriangles;
    bool _readbackPending;
    bool _disposed;

    void Initialize((int, int, int) dims, int budget, ComputeShader compute)
    {
        _grids = dims;
        _triangleBudget = budget;
        _compute = compute;

        AllocateBuffers();
        AllocateMesh(3 * _triangleBudget);
    }

    void ReleaseAll()
    {
        _disposed = true;
        ReleaseBuffers();
        ReleaseMesh();
    }

    void RunCompute(ComputeBuffer voxels, float target, float scale)
    {
        _counterBuffer.SetCounterValue(0);

        // Isosurface reconstruction
        _compute.SetInts(DimsId, _grids);
        _compute.SetInt(MaxTriangleId, _triangleBudget);
        _compute.SetFloat(ScaleId, scale);
        _compute.SetFloat(IsovalueId, target);
        _compute.SetBuffer(0, TriangleTableId, _triangleTable);
        _compute.SetBuffer(0, VoxelsId, voxels);
        _compute.SetBuffer(0, VertexBufferId, _vertexBuffer);
        _compute.SetBuffer(0, IndexBufferId, _indexBuffer);
        _compute.SetBuffer(0, CounterId, _counterBuffer);
        _compute.DispatchThreads(0, _grids);

        // Snapshot the produced-triangle count before the clear kernel
        // advances the counter, and read it back when the GPU is done.
        // Only copy when issuing a fresh request: writing the destination
        // while a readback is still in flight makes that readback fail.
        if (!_readbackPending)
        {
            ComputeBuffer.CopyCount(_counterBuffer, _countCopyBuffer, 0);
            _readbackPending = true;
            AsyncGPUReadback.Request(_countCopyBuffer, OnReadback);
        }

        // Clear the slack between the produced count and the visible range.
        _compute.SetInt(MaxTriangleId, _submeshTriangles);
        _compute.SetBuffer(1, VertexBufferId, _vertexBuffer);
        _compute.SetBuffer(1, IndexBufferId, _indexBuffer);
        _compute.SetBuffer(1, CounterId, _counterBuffer);
        _compute.DispatchThreads(1, 1024, 1, 1);

        // Bounding box
        var ext = new Vector3(_grids.x, _grids.y, _grids.z) * scale;
        _mesh.bounds = new Bounds(Vector3.zero, ext);
    }

    // Called on the main thread the frame the readback lands. Polling the
    // request later would not work: completed request data is discarded
    // after a frame, and RunCompute may not run every frame.
    void OnReadback(AsyncGPUReadbackRequest request)
    {
        _readbackPending = false;
        if (_disposed || request.hasError) return;

        LastTriangleCount =
          Mathf.Min((int)request.GetData<uint>()[0], _triangleBudget);

        var count = LastTriangleCount;
        var wanted = Mathf.Min(_triangleBudget,
          Mathf.CeilToInt(count * (1 + MarginRatio)) + MarginTriangles);

        // Grow once the remaining headroom drops below half the margin,
        // rather than on every new high, so the margin actually absorbs
        // growth while still covering a count that is a few frames stale.
        var reserve = Mathf.CeilToInt(count * (1 + MarginRatio / 2))
                      + MarginTriangles / 2;
        var grow = reserve > _submeshTriangles &&
                   _submeshTriangles < _triangleBudget;
        var shrink = wanted < _submeshTriangles * ShrinkRatio;
        if (grow || shrink) SetSubmeshTriangles(wanted);
    }

    #endregion

    #region Compute buffer objects

    ComputeBuffer _triangleTable;
    ComputeBuffer _counterBuffer;
    ComputeBuffer _countCopyBuffer;

    void AllocateBuffers()
    {
        // Marching cubes triangle table
        _triangleTable = new ComputeBuffer(256, sizeof(ulong));
        _triangleTable.SetData(PrecalculatedData.TriangleTable);

        // Buffer for triangle counting
        _counterBuffer = new ComputeBuffer(1, 4, ComputeBufferType.Counter);

        // Destination for CopyCount + async readback
        _countCopyBuffer = new ComputeBuffer(1, 4, ComputeBufferType.Raw);
    }

    void ReleaseBuffers()
    {
        _triangleTable.Dispose();
        _counterBuffer.Dispose();
        _countCopyBuffer.Dispose();
    }

    #endregion

    #region Mesh objects

    Mesh _mesh;
    GraphicsBuffer _vertexBuffer;
    GraphicsBuffer _indexBuffer;

    void AllocateMesh(int vertexCount)
    {
        _mesh = new Mesh();

        // We want GraphicsBuffer access as Raw (ByteAddress) buffers.
        _mesh.indexBufferTarget |= GraphicsBuffer.Target.Raw;
        _mesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw;

        // Vertex position: float32 x 3
        var vp = new VertexAttributeDescriptor
          (VertexAttribute.Position, VertexAttributeFormat.Float32, 3);

        // Vertex normal: float32 x 3
        var vn = new VertexAttributeDescriptor
          (VertexAttribute.Normal, VertexAttributeFormat.Float32, 3);

        // Vertex/index buffer formats
        _mesh.SetVertexBufferParams(vertexCount, vp, vn);
        _mesh.SetIndexBufferParams(vertexCount, IndexFormat.UInt32);

        // Start with the full budget exposed; the first readback trims it.
        SetSubmeshTriangles(_triangleBudget);
    }

    void SetSubmeshTriangles(int triangles)
    {
        _submeshTriangles = Mathf.Clamp(triangles, 0, _triangleBudget);
        _mesh.SetSubMesh(0, new SubMeshDescriptor(0, _submeshTriangles * 3),
                         MeshUpdateFlags.DontRecalculateBounds |
                         MeshUpdateFlags.DontValidateIndices |
                         MeshUpdateFlags.DontNotifyMeshUsers |
                         MeshUpdateFlags.DontResetBoneBounds);

        // GraphicsBuffer handles can be invalidated by mesh edits; re-acquire.
        _vertexBuffer?.Dispose();
        _indexBuffer?.Dispose();
        _vertexBuffer = _mesh.GetVertexBuffer(0);
        _indexBuffer = _mesh.GetIndexBuffer();
    }

    void ReleaseMesh()
    {
        _vertexBuffer.Dispose();
        _indexBuffer.Dispose();
        Object.Destroy(_mesh);
    }

    #endregion
}

} // namespace MarchingCubes
