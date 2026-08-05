"""Conservative cosmetic-category suggestions for prototype UI integration."""

from __future__ import annotations

import re

from src.api.schemas import ProductRecommendation, Questionnaire


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)


def _is_minor_age_range(age_range: str) -> bool:
    numbers = [int(value) for value in re.findall(r"\d+", age_range)]
    return bool(numbers) and max(numbers) < 18


def build_recommendations(
    questionnaire: Questionnaire,
    detection_count: int,
    mean_confidence: float,
    dominant_region: str,
) -> list[ProductRecommendation]:
    """Use questionnaire and model summary to choose non-treatment categories."""
    skin_type = questionnaire.skin_type.casefold()
    concerns = " ".join(questionnaire.concerns).casefold()
    goal = questionnaire.goal.casefold()

    cleanser_focus = "gentle daily cleanser"
    if _contains_any(skin_type, ("oily", "combination")):
        cleanser_focus = "gentle lightweight cleanser"
    elif _contains_any(skin_type, ("dry", "sensitive")):
        cleanser_focus = "gentle non-stripping cleanser"

    serum_focus = "simple hydrating cosmetic serum"
    if _contains_any(f"{concerns} {goal}", ("oil", "shine", "breakout", "acne")):
        serum_focus = "lightweight oil-balancing cosmetic serum"
    elif _contains_any(f"{concerns} {goal}", ("dry", "dehydrat", "barrier", "sensitive", "calm")):
        serum_focus = "hydrating or soothing cosmetic serum"

    moisturiser_focus = "lightweight moisturiser"
    if _contains_any(skin_type, ("dry", "sensitive")):
        moisturiser_focus = "barrier-supporting moisturiser"
    elif _contains_any(skin_type, ("oily", "combination")):
        moisturiser_focus = "lightweight non-greasy moisturiser"

    recommendations = [
        ProductRecommendation(
            category="cleanser",
            focus=cleanser_focus,
            rationale=f"Selected from the reported {questionnaire.skin_type} skin type and routine goal.",
        ),
        ProductRecommendation(
            category="serum",
            focus=serum_focus,
            rationale="Selected from the self-reported concerns and goal, not inferred from the image.",
        ),
        ProductRecommendation(
            category="moisturiser",
            focus=moisturiser_focus,
            rationale=f"A conservative category matched to the reported skin type and age range ({questionnaire.age_range}).",
        ),
        ProductRecommendation(
            category="sunscreen",
            focus="broad-spectrum daily sunscreen",
            rationale="A general cosmetic routine category; no condition or treatment claim is made.",
        ),
    ]

    model_supports_spot_category = detection_count > 0 and mean_confidence >= 0.25
    questionnaire_supports_spot_category = _contains_any(
        f"{concerns} {goal}", ("breakout", "acne", "spot", "blemish")
    )
    if model_supports_spot_category or questionnaire_supports_spot_category:
        model_context = (
            f"The model marked {detection_count} candidate region(s), concentrated approximately in {dominant_region.replace('_', ' ')}."
            if model_supports_spot_category
            else "This category is based only on the self-reported concern or goal."
        )
        recommendations.append(
            ProductRecommendation(
                category="optional spot care",
                focus="non-prescription cosmetic spot-care category",
                rationale=f"{model_context} This is not treatment advice.",
            )
        )

    if not _is_minor_age_range(questionnaire.age_range) and _contains_any(
        f"{concerns} {goal}", ("wellness", "nutrition", "diet", "supplement")
    ):
        recommendations.append(
            ProductRecommendation(
                category="optional supplement category",
                focus="general wellness supplement category",
                rationale="Included only from the reported goal; check suitability with a qualified professional.",
            )
        )
    return recommendations

