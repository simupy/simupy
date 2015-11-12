"""
Follow Betsch, Diebel to be formal and consistent
Try to find good process and examples of using quaternions
to formulate dynamics and relating to Euler stuff
"""

def quaternion_conjugate(q):
    return sp.ImmutableMatrix([q[0]] + (-q)[1:])

# Setup and basic definitions
def cross_product_matrix(x):
    return sp.ImmutableMatrix([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
    
def quaternion_matrix(q):
    out = zeros(4,4)
    out[0] = q[0]
    out[0,1:] = -q[1:,:].T
    out[1:,0] = q[1:,:]
    out[1:,1:] = q[0]*eye(3)+cross_product_matrix(q[1:,0])
    return sp.ImmutableMatrix(out)

def quaternionrate_eulerrate_matrix(eulerrates):
    return sp.Rational(1,2)*sp.ImmutableMatrix([
        [0, -eulerrates[0], -eulerrates[1], -eulerrates[2]],
        [eulerrates[0], 0, eulerrates[2], -eulerrates[1]],
        [eulerrates[1], -eulerrates[2], 0, eulerrates[0]],
        [eulerrates[2], eulerrates[1], -eulerrates[0], 0]
    ])
    
def angle_def_from_rot_matrices(eul_mat, quat_mat, angle):
    x = sp.Wild('x')
    definitions = []
    for i in range(len(eul_mat)):
        for j in range(i):
            sin_match = eul_mat[i].match(x*sin(angle))
            cos_match = eul_mat[j].match(x*cos(angle))
            if sin_match is not None and cos_match is not None and (sin_match[x] == cos_match[x] or sin_match[x] == -cos_match[x]):
                definitions.append( atan2(quat_mat[i], quat_mat[j]) )
            if sin_match is not None and sin_match[x].is_number:
                definitions.append( asin(sin_match[x]*quat_mat[i]) )
            if cos_match is not None and cos_match[x].is_number:
                definitions.append( acos(cos_match[x]*quat_mat[j]) )
    if definitions.count(definitions[0]) == len(definitions):
        return definitions[0]
    else:
        return definitions