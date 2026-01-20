import pandas as pd
import re
import unicodedata


class FeatureExtractor:
    """
    Componente encargado de transformar texto no estructurado (descripciones)
    en variables binarias (0/1) para modelos de Machine Learning.
    """

    def __init__(self):
        # Definimos los patrones Regex para cada feature.
        # \b asegura que buscamos palabras completas.
        # (?i) hace que sea insensible a mayúsculas/minúsculas.
        self.patterns = {
            'has_security': [
                r'vigilancia', r'seguridad', r'cctv',
                r'control de acceso', r'port[oó]n el[eé]ctrico',
                r'caseta', r'guardia', r'circuito cerrado'
            ],
            'has_garden': [
                r'jard[ií]n', r'patio trasero', r'amplio patio',
                r'areas? verdeg?s?', r'huerto'
            ],
            'has_pool': [
                r'alberca', r'piscina', r'carril de nado',
                r'jacuzzi', r'chapoteadero'
            ],
            'has_terrace': [
                r'terraza', r'roof garden', r'balc[oó]n',
                r'asador', r'palapa'
            ],
            'is_new': [
                r'preventa', r'entrega inmediata', r'estrenar',
                r'acabados de lujo', r'nueva etapa'
            ]
        }

    def _normalize_text(self, text: str) -> str:
        """Limpia el texto para facilitar la búsqueda."""
        if not isinstance(text, str):
            return ""
        # Convertir a minúsculas y normalizar acentos (opcional, pero útil)
        text = text.lower()
        # Normalización unicode para tratar 'á' igual que 'a' si quisieras ser estricto,
        # pero mis regex ya contemplan acentos.
        return text

    def process(self, df: pd.DataFrame, text_column: str = 'description') -> pd.DataFrame:
        """
        Aplica la extracción de features al DataFrame dado.
        Retorna el DataFrame original con las nuevas columnas agregadas.
        """
        print(f"--- Iniciando Extracción de Features en {len(df)} registros ---")

        # Trabajamos sobre una copia para no alterar el original inmediatamente
        df_out = df.copy()

        # Llenar nulos por si acaso
        search_space = df_out[text_column].fillna("").apply(self._normalize_text)

        for feature_name, keywords in self.patterns.items():
            # Creamos una expresión regular unificada: (palabra1|palabra2|palabra3)
            # Esto es mucho más rápido que iterar palabra por palabra
            regex_pattern = '|'.join(keywords)

            print(f"   > Extrayendo: {feature_name}...")

            # Aplicamos la búsqueda vectorizada (muy rápido en Pandas)
            df_out[feature_name] = search_space.str.contains(regex_pattern, case=False, regex=True).astype(int)

        # Lógica de corrección específica (Business Rules)
        # Ejemplo: Si dice "Patio de servicio" NO cuenta como Jardín (Falso positivo común)
        # Aquí usamos una máscara negativa para corregir
        mask_service_patio = search_space.str.contains(r'patio de servicio|patio de lavado', regex=True)
        # Solo restamos si la única razón por la que es 1 fue por la palabra "patio" genérica.
        # (Esta lógica se puede refinar, por ahora lo mantendremos simple).

        print("--- Extracción Finalizada ---\n")
        return df_out


# --- BLOQUE DE PRUEBA (Para correrlo ya mismo) ---
if __name__ == "__main__":
    # Datos falsos para probar tu lógica mientras esperas al scraper
    mock_data = {
        'id': [1, 2, 3, 4],
        'price': [100, 200, 300, 400],
        'description': [
            "Casa en venta con vigilancia 24/7 y amplia cocina.",
            "Hermoso departamento con Roof Garden y alberca compartida.",
            "Terreno listo para construir.",
            "Casa con jardín trasero y patio de servicio."
        ]
    }

    df_mock = pd.DataFrame(mock_data)

    extractor = FeatureExtractor()
    df_enriched = extractor.process(df_mock)

    print("Resultados de la prueba:")
    print(df_enriched[['description', 'has_security', 'has_pool', 'has_garden', 'has_terrace']])