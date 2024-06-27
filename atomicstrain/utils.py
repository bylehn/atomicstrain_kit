def create_selections(ref, defm, residue_numbers, protein_ca, R):
    selections = []
    for resid in residue_numbers:
        selection_str = f"({protein_ca} and around {R} (resid {resid} and name CA))"
        center_str = f"resid {resid} and name CA"

        ref_selection = ref.select_atoms(selection_str)
        ref_center = ref.select_atoms(center_str)

        ref_resids = ref_selection.resids
        defm_selection_str = f"(name CA and resid {' '.join(map(str, ref_resids))})"
        defm_center_str = f"resid {resid} and name CA" 

        defm_selection = defm.select_atoms(defm_selection_str)
        defm_center = defm.select_atoms(defm_center_str)

        selections.append(((ref_selection, ref_center), (defm_selection, defm_center)))
    return selections