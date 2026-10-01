using UnityEngine;
using System.Collections.Generic;

namespace MarchingCubes
{
    [System.Serializable]
    public class Metaball
    {
        public Vector3 Position;
        public float Radius;
    }

    //
    // GPU work only runs when something changed (a metaball moved, resized,
    // was added/removed, or a grid setting changed). The current metaball
    // state is compared against the last uploaded copy each frame, so a
    // static scene costs nothing beyond that CPU-side comparison.
    //
    sealed class MetaballsVisualizer : MonoBehaviour
    {
        #region Editable attributes

        [SerializeField] Vector3Int _dimensions = new Vector3Int(64, 32, 64);
        [SerializeField] float _gridScale = 4.0f / 64;
        [SerializeField] int _triangleBudget = 65536;
        [SerializeField] float _targetValue = 0.26f;

        [SerializeField] List<Metaball> metaballs = new List<Metaball>();

        #endregion

        #region Project asset references

        [SerializeField, HideInInspector] ComputeShader _volumeCompute = null;
        [SerializeField, HideInInspector] ComputeShader _builderCompute = null;

        #endregion

        #region Private members

        static readonly int DimsId = Shader.PropertyToID("Dims");
        static readonly int ScaleId = Shader.PropertyToID("Scale");
        static readonly int VoxelsId = Shader.PropertyToID("Voxels");
        static readonly int MetaballCentersId = Shader.PropertyToID("MetaballCenters");
        static readonly int MetaballRadiiId = Shader.PropertyToID("MetaballRadii");

        int VoxelCount => _dimensions.x * _dimensions.y * _dimensions.z;

        // ComputeBuffer can't be empty, so with no metaballs the buffers keep
        // one zero-radius entry, which contributes nothing to the field.
        int BufferCount => Mathf.Max(1, metaballs.Count);

        ComputeBuffer _voxelBuffer;
        ComputeBuffer _positionsBuffer;
        ComputeBuffer _radiiBuffer;
        MeshBuilder _builder;

        // Last uploaded metaball state; reused every upload so there are no
        // per-frame allocations, and compared against to detect changes.
        Vector3[] _positionsScratch;
        float[] _radiiScratch;
        int _uploadedCount = -1;

        // Settings the last build used.
        bool _fieldDirty = true;
        float _builtGridScale;
        float _builtTargetValue;
        int _builtSubmeshTriangles;

        #endregion

        #region MonoBehaviour implementation

        void Start()
        {
            _voxelBuffer = new ComputeBuffer(VoxelCount, sizeof(float));
            InitializeMetaballBuffers();
            _builder = new MeshBuilder(_dimensions, _triangleBudget, _builderCompute);
            GetComponent<MeshFilter>().sharedMesh = _builder.Mesh;
        }

        void OnDestroy()
        {
            _voxelBuffer?.Dispose();
            ReleaseMetaballBuffers();
            _builder.Dispose();
        }

        void Update()
        {
            if (_gridScale != _builtGridScale || _targetValue != _builtTargetValue)
                _fieldDirty = true;

            // A grown submesh exposes a range the clear kernel has not
            // zeroed yet; rebuild once so it holds degenerate triangles.
            // A shrink only hides triangles, so it just needs recording.
            var submesh = _builder.SubmeshTriangleCount;
            if (submesh > _builtSubmeshTriangles) _fieldDirty = true;
            else _builtSubmeshTriangles = submesh;

            if (MetaballsChanged()) _fieldDirty = true;

            if (!_fieldDirty) return;

            UploadMetaballBuffers();

            _volumeCompute.SetInts(DimsId, _dimensions);
            _volumeCompute.SetFloat(ScaleId, _gridScale);
            _volumeCompute.SetBuffer(0, VoxelsId, _voxelBuffer);
            _volumeCompute.SetBuffer(0, MetaballCentersId, _positionsBuffer);
            _volumeCompute.SetBuffer(0, MetaballRadiiId, _radiiBuffer);
            _volumeCompute.DispatchThreads(0, _dimensions);

            // Isosurface reconstruction
            _builder.BuildIsosurface(_voxelBuffer, _targetValue, _gridScale);

            _builtGridScale = _gridScale;
            _builtTargetValue = _targetValue;
            _builtSubmeshTriangles = _builder.SubmeshTriangleCount;
            _fieldDirty = false;
        }

        #endregion

        #region Helper Methods

        void InitializeMetaballBuffers()
        {
            // Create buffers
            _positionsBuffer = new ComputeBuffer(BufferCount, sizeof(float) * 3);
            _radiiBuffer = new ComputeBuffer(BufferCount, sizeof(float));
            _positionsScratch = new Vector3[BufferCount];
            _radiiScratch = new float[BufferCount];
        }

        bool MetaballsChanged()
        {
            if (_uploadedCount != metaballs.Count) return true;

            for (int i = 0; i < metaballs.Count; i++)
            {
                // Exact compare; Vector3 != ignores moves under ~1e-5.
                if (!_positionsScratch[i].Equals(metaballs[i].Position)) return true;
                if (_radiiScratch[i] != metaballs[i].Radius) return true;
            }

            return false;
        }

        void UploadMetaballBuffers()
        {
            // Ensure buffers match the metaball count
            if (_positionsBuffer.count != BufferCount)
            {
                ReleaseMetaballBuffers();
                InitializeMetaballBuffers();
            }

            // Update positions and radii arrays (padding entries get radius 0)
            for (int i = 0; i < BufferCount; i++)
            {
                var active = i < metaballs.Count;
                _positionsScratch[i] = active ? metaballs[i].Position : Vector3.zero;
                _radiiScratch[i] = active ? metaballs[i].Radius : 0;
            }

            _positionsBuffer.SetData(_positionsScratch);
            _radiiBuffer.SetData(_radiiScratch);
            _uploadedCount = metaballs.Count;
        }

        void ReleaseMetaballBuffers()
        {
            _positionsBuffer?.Dispose();
            _radiiBuffer?.Dispose();
        }

        #endregion
    }
}
