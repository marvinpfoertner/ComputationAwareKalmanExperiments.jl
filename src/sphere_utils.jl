using GeometryBasics

function gcs_to_cartesian(λ, θ; r=1.0)
    return [
        r * cos(λ) * cos(θ),
        r * sin(λ) * cos(θ),
        r * sin(θ),
    ]
end

function sphere_mesh(λs, θs; r=1.0)
    points_mesh = [
        Point3f(gcs_to_cartesian(λ, θ, r=r))
        for θ in θs
        for λ in λs
    ]
    faces_mesh = decompose(
        QuadFace{GLIndex},
        Tesselation(
            Rect(0, 0, 1, 1),
            (size(λs, 1), size(θs, 1)),
        )
    )
    normals_mesh = normalize.(points_mesh)
    uv_mesh = [
        Vec2f(
            (j - 1) / (size(θs, 1) - 1),
            1 - (i - 1) / ((size(λs, 1) - 1)),
        )
        for j in 1:size(θs, 1)
        for i in 1:size(λs, 1)
    ]

    return GeometryBasics.Mesh(
        meta(points_mesh; uv=uv_mesh, normals=normals_mesh),
        faces_mesh
    )
end