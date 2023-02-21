from napari_amdtrk import reader_function

def test_read_basic():
    layers = reader_function('simple')
    assert len(layers) == 4, 'Failed to load basic sample data.'
    pass

def test_read_full():
    layers = reader_function('full')
    assert len(layers) == 4, 'Failed to load complete sample data.'
    pass
