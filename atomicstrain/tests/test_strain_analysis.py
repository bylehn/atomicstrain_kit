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
    residue_numbers = list(range(1, 11))
    min_neighbors = 3

    print(f"Debug: ref atoms: {len(ref.atoms)}, ref residues: {len(ref.residues)}")
    print(f"Debug: defm atoms: {len(defm.atoms)}, defm residues: {len(defm.residues)}")
    print(f"Debug: ref CA atoms: {len(ref.select_atoms('name CA'))}")
    print(f"Debug: defm CA atoms: {len(defm.select_atoms('name CA'))}")
    
    selections = create_selections(ref, defm, residue_numbers, min_neighbors)
    
    assert len(selections) > 0, "No selections were created"
    for (ref_sel, ref_center), (defm_sel, defm_center) in selections:
        assert len(ref_sel) > min_neighbors, f"Not enough atoms selected for center {ref_center.resids[0]}"
        assert len(defm_sel) > min_neighbors, f"Not enough atoms selected for center {defm_center.resids[0]}"
        assert len(ref_sel) == len(defm_sel), f"Mismatched selection sizes for center {ref_center.resids[0]}"

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
    assert jnp.all(jnp.diff(principal_strains) <= 0)  # Check if sorted in descending order

def test_strain_analysis_initialization(mock_universes):
    """Test the initialization of StrainAnalysis."""
    ref, defm = mock_universes
    residue_numbers = list(range(1, 11))
    min_neighbors = 3

    analysis = StrainAnalysis(ref, defm, residue_numbers, min_neighbors)

    assert analysis.ref == ref
    assert analysis.defm == defm
    assert analysis.residue_numbers == residue_numbers
    assert analysis.min_neighbors == min_neighbors
    assert len(analysis.selections) == len(residue_numbers)

def test_strain_analysis_run(mock_universes):
    """Test the full run of StrainAnalysis."""
    ref, defm = mock_universes
    residue_numbers = list(range(1, 11))
    min_neighbors = 3

    analysis = StrainAnalysis(ref, defm, residue_numbers, min_neighbors)
    analysis.run()

    assert hasattr(analysis.results, 'shear_strains')
    assert hasattr(analysis.results, 'principal_strains')
    assert analysis.results.shear_strains.shape[0] == len(defm.trajectory)
    assert analysis.results.shear_strains.shape[1] == len(residue_numbers)
    assert analysis.results.principal_strains.shape[0] == len(defm.trajectory)
    assert analysis.results.principal_strains.shape[1] == len(residue_numbers)
    assert analysis.results.principal_strains.shape[2] == 3  # 3 principal strains per residue

if __name__ == "__main__":
    pytest.main()