def generate_explanation(features):
    """
    Generate human-readable explanations
    based on extracted audio features.

    Input:
    - features: dict from extract_features()

    Output:
    - list of explanation strings
    """

    explanations = []

    # Thresholds (heuristic, hackathon-safe)
    if features.get("pitch_std", 0) < 10:
        explanations.append(
            "Unnaturally stable pitch patterns detected"
        )

    if features.get("spectral_flux_std", 0) < 0.05:
        explanations.append(
            "Over-smooth spectral transitions"
        )

    if features.get("energy_std", 0) < 0.02:
        explanations.append(
            "Reduced temporal energy variation"
        )

    if features.get("hnr", 0) > 18:
        explanations.append(
            "Excessively clean harmonic structure"
        )

    # Fallback explanation
    if not explanations:
        explanations.append(
            "Speech characteristics resemble natural human voice patterns"
        )

    return explanations
