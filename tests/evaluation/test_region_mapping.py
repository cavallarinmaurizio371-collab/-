import pytest

from evaluation.coordinate_adapter import nearest_region


@pytest.fixture
def targets():
    return {
        "LEFT_UP":(-150,100),"UP":(0,100),"RIGHT_UP":(150,100),
        "LEFT":(-150,0),"CENTER":(0,0),"RIGHT":(150,0),
        "LEFT_DOWN":(-150,-100),"DOWN":(0,-100),"RIGHT_DOWN":(150,-100),
    }


@pytest.mark.parametrize("point,expected",[
    ((-140,90),"LEFT_UP"),((5,85),"UP"),((140,90),"RIGHT_UP"),
    ((-130,5),"LEFT"),((3,-2),"CENTER"),((130,-5),"RIGHT"),
    ((-140,-90),"LEFT_DOWN"),((0,-85),"DOWN"),((145,-90),"RIGHT_DOWN"),
])
def test_known_points_map_to_physical_regions(targets,point,expected):
    assert nearest_region(*point,targets)==expected

