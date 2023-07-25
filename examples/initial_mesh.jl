using QuadMeshGame
using RandomQuadMesh
using PlotQuadMesh

RQ = RandomQuadMesh
QM = QuadMeshGame
PQ = PlotQuadMesh

boundary = [0.0 0.2 1.0 1.0 1.0 0.8 0.0 0.0
            0.0 0.0 0.0 0.8 1.0 1.0 1.0 0.7]

mesh = RQ.tri_mesh(boundary, hmax=0.6)
q, t = RQ.match_tri2quad(mesh)
mesh = RQ.triquad_refine(mesh.p, q, t)
mesh = QM.QuadMesh(mesh.p, mesh.t)
QM.averagesmoothing!(mesh)
# mesh = RQ.quad_mesh(boundary, algorithm="matching")
fig, ax = PQ.plot_mesh(
    QM.active_vertex_coordinates(mesh),
    QM.active_quad_connectivity(mesh)
)
fig.savefig("examples/figures/irregular.png")