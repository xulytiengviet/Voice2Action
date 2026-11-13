import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from v2a import spec, safety, FS

def test_spectrogram():
    signal=np.sin(2*np.pi*440*np.arange(FS)/FS).astype("float32")*.4
    bands=spec(signal, FS)
    assert bands.shape[0]==32 and bands.shape[1]>50 and np.isfinite(bands).all()

def test_silence():
    with pytest.raises(ValueError): spec(np.zeros(FS,dtype="float32"), FS)

def test_unknown_action():
    assert safety("unknown",.99,.9)["decision"]=="REJECT"

def test_confidence():
    assert safety("left",.1,.1)["decision"]=="REJECT"

def test_estop():
    assert safety("forward",.99,.9,{"x":0,"y":0,"heading":0,"estop":True})["decision"]=="REJECT"

def test_bounds_and_keepout():
    assert safety("forward",.99,.9,{"x":.95,"y":0,"heading":0})["reason"]=="boundary"
    assert safety("forward",.99,.9,{"x":.4,"y":.5,"heading":0})["reason"]=="keepout"

def test_wrong_but_approved_action():
    assert safety("right",.99,.9)["decision"]=="ALLOW"
