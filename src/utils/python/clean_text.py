import re


class AddressCleaner:
    """
    Service class for normalizing real estate address strings.
    Implements noise reduction, macro-location pruning, and structural deduplication.
    Optimized to preserve proper names starting with articles (e.g., 'La Vista').
    """

    # Marketing patterns and common OCR/Typo errors to strip
    NOISE_PATTERNS = [
        r"venta de casa en", r"casa en venta", r"en venta",
        r"venta", r"preventa", r"remate", r"oportunidad",
        r"fraccionamiento", r"residencial", r"condominio",
        r"lotes?", r"terrenos?", r"departamentos?", r"casas?",
        r"\bnueva\b", r"\bnuevo\b",  # Removes adjectives like "Nueva en..."
        r"fraccionamient[o0]"  # Handles common typos like 'fraccionamient0'
    ]

    @staticmethod
    def clean(raw_address: str) -> str:
        """
        Applies a pipeline of regex substitutions to standardize the address.
        """
        if not isinstance(raw_address, str) or len(raw_address) < 3:
            return ""

        cleaned = raw_address.lower()

        # 1. Strip Marketing Noise
        for pattern in AddressCleaner.NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 2. Prune Macro-Locations (City/State names)
        # Prevents recursion like "Juriquilla Queretaro Queretaro" during API calls.
        cleaned = re.sub(r'\b(quer[ée]taro|m[ée]xico|qro)\b', '', cleaned)

        # 3. Structural Deduplication
        # Fixes "Loma Dorada, Loma Dorada" -> "Loma Dorada"
        cleaned = re.sub(r'\b(.+?)(?:[\s,]+)\1\b', r'\1', cleaned)

        # 4. Remove Orphan Prepositions (Artifact Cleanup)
        # ONLY remove 'en' or 'de' if they are left at the start.
        # PROTECT 'el', 'la', 'los', 'las' as they are often part of proper names (e.g., 'El Refugio').
        cleaned = re.sub(r'^\s*(?:en|de)\b\s*', '', cleaned)

        # 5. Final Formatting (Remove special chars except Spanish accents)
        cleaned = re.sub(r'[^a-z0-9\s,áéíóúñ]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'^,+,*|,*,$', '', cleaned)

        return cleaned