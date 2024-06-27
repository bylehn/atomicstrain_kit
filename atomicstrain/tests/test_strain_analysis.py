import pytest
import numpy as np
import jax.numpy as jnp
import MDAnalysis as mda
from atomicstrain.analysis import StrainAnalysis
from atomicstrain.compute import compute_strain_tensor, compute_principal_strains_and_shear
from atomicstrain.utils import create_selections
from .utils import make_Universe

@pytest.fixture
def mock_universes():
    """Create mock reference and deformed universes for testing."""
    ref = make_Universe(extras=('masses',), size=(100, 10, 1), n_frames=1)
    defm = make_Universe(extras=('masses',), size=(100, 10, 1), n_frames=10)
    
    # Set some fake coordinates
    ref.atoms.positions = np.random.rand(100, 3) * 10
    defm.trajectory[0].positions = ref.atoms.positions + np.random.rand(100, 3) * 0.1
    
    return ref, defm

def test_create_selections(mock_universes):
    """Test the create_selections function."""
    ref, defm = mock_universes
    residue_numbers = [1, 2, 3]
    protein_ca = 'name CA'
    R = 5.0
    
    selections = create_selections(ref, defm, residue_numbers, protein_ca, R)
    
    assert len(selections) == len(residue_numbers)
    for ((ref_sel, ref_center), (defm_sel, defm_center)) in selections:
        assert isinstance(ref_sel, mda.AtomGroup)
        assert isinstance(ref_center, mda.AtomGroup)
        assert isinstance(defm_sel, mda.AtomGroup)
        assert isinstance(defm_center, mda.AtomGroup)

def test_compute_strain_tensor():
    """Test the compute_strain_tensor function."""
    Am = np.random.rand(10, 3)
    Bm = Am + np.random.rand(10, 3) * 0.1
    
    Q = compute_strain_tensor(Am, Bm)
    
    assert Q.shape == (3, 3)
    assert np.allclose(Q, Q.T)  # Strain tensor should be symmetric

def test_compute_principal_strains_and_shear():
    """Test the compute_principal_strains_and_shear function."""
    Q = np.random.rand(3, 3)
    Q = (Q + Q.T) / 2  # Make it symmetric
    
    shear, principal_strains = compute_principal_strains_and_shear(Q)
    
    assert isinstance(shear, jnp.ndarray)
    assert shear.shape == ()  # Check if it's a scalar
    sorted_strains = jnp.sort(principal_strains)[::-1]  # Sort in descending order
    assert jnp.allclose(principal_strains, sorted_strains, atol=1e-5)

def test_strain_analysis_initialization(mock_universes):
    """Test the initialization of StrainAnalysis."""
    ref, defm = mock_universes
    residue_numbers = list(range(1, 11))
    protein_ca = 'name CA'
    R = 5.0
    
    analysis = StrainAnalysis(ref, defm, residue_numbers, protein_ca, R)
    
    assert analysis.ref == ref
    assert analysis.defm == defm
    assert analysis.residue_numbers == residue_numbers
    assert analysis.protein_ca == protein_ca
    assert analysis.R == R
    assert len(analysis.selections) == len(residue_numbers)

def test_strain_analysis_run(mock_universes):
    """Test the full run of StrainAnalysis."""
    ref, defm = mock_universes
    residue_numbers = list(range(1, 11))
    protein_ca = 'name CA'
    R = 5.0
    
    analysis = StrainAnalysis(ref, defm, residue_numbers, protein_ca, R)
    analysis.run()
    
    assert hasattr(analysis.results, 'shear_strains')
    assert hasattr(analysis.results, 'principal_strains')
    assert hasattr(analysis.results, 'avg_shear_strains')
    assert hasattr(analysis.results, 'avg_principal_strains')
    
    assert analysis.results.shear_strains.shape == (10, 10)  # 10 frames, 10 residues
    assert analysis.results.principal_strains.shape == (10, 10, 3)  # 10 frames, 10 residues, 3 principal strains
    assert analysis.results.avg_shear_strains.shape == (10,)  # 10 residues
    assert analysis.results.avg_principal_strains.shape == (10, 3)  # 10 residues, 3 principal strains

if __name__ == "__main__":
    pytest.main()