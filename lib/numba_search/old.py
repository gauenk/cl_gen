

def get_qwindow(q_index,Q,W,R=5):
    # left = max([q_index - 100,0])
    # right = min([q_index + 100,Q])
    qRow = q_index // W
    qCol = q_index % W
    
    c_left = max([qCol - R,0])
    c_right = min([qCol + R,Q])
    cgrid = torch.arange(c_left,c_right)

    indices = []
    for rowOffset in range(-R-1,R,1):
        if rowOffset < 0:
            row = max([qRow + rowOffset,0])
        elif rowOffset > 0:
            row = min([qRow + rowOffset,Q])            
        print(row,rowOffset)
        indices.append(cgrid + row * W)
    indices = torch.cat(indices,dim=0)

    r_top = max([qRow - R ,0])
    r_bottom = max([qRow + R,Q])

    box = [r_top,c_left,2*R,2*R]
    indicesv2 = tvF.crop(index_grid,r_top,c_left,2*R,2*R)
    if len(indices) == (2*R)**2:
        print(q_index,indices.reshape(2*R,2*R),indicesv2)
    return indices
