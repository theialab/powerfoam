import warp as wp


@wp.func
def C0() -> float:
    return 0.28209479177387814


@wp.func
def C1() -> float:
    return 0.4886025119029199


@wp.func
def C2(i: int) -> float:
    assert i >= 0 and i < 5, "Invalid index for C2"
    if i == 0:
        return 1.0925484305920792
    elif i == 1:
        return -1.0925484305920792
    elif i == 2:
        return 0.31539156525252005
    elif i == 3:
        return -1.0925484305920792
    elif i == 4:
        return 0.5462742152960396
    else:
        return 0.0  # This should never be reached due to the assert


@wp.func
def C3(i: int) -> float:
    assert i >= 0 and i < 7, "Invalid index for C3"
    if i == 0:
        return -0.5900435899266435
    elif i == 1:
        return 2.890611442640554
    elif i == 2:
        return -0.4570457994644658
    elif i == 3:
        return 0.3731763325901154
    elif i == 4:
        return -0.4570457994644658
    elif i == 5:
        return 1.445305721320277
    elif i == 6:
        return -0.5900435899266435
    else:
        return 0.0  # This should never be reached due to the assert


@wp.func
def C4(i: int) -> float:
    assert i >= 0 and i < 9, "Invalid index for C4"
    if i == 0:
        return 2.5033429417967046
    elif i == 1:
        return -1.7701307697799304
    elif i == 2:
        return 0.9461746957575601
    elif i == 3:
        return -0.6690465435572892
    elif i == 4:
        return 0.10578554691520431
    elif i == 5:
        return -0.6690465435572892
    elif i == 6:
        return 0.47308734787878004
    elif i == 7:
        return -1.7701307697799304
    elif i == 8:
        return 0.6258357354491761
    else:
        return 0.0  # This should never be reached due to the assert
