import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import argparse

def download_resources():
    """Descarga los recursos necesarios de NLTK (solo 1st time)"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        print("Advertencia: No se pudieron descargar algunos recursos de NLTK")


class VectorRepresentation:
    """Genera representaciones vectoriales de reseñas"""
    
    def __init__(self, use_stemming=False, use_lemmatization=True, 
                 remove_stopwords=True, max_features=5000):
        """
        Inicializa el generador de representaciones vectoriales
        
        Args:
            use_stemming (bool): Aplicar stemming
            use_lemmatization (bool): Aplicar lemmatization
            remove_stopwords (bool): Eliminar stopwords
            max_features (int): Número máximo de características para TF-IDF
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.max_features = max_features
        
        # Inicializar herramientas
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Vectorizadores (se entrenarán con el corpus)
        self.vectorizers = {}
        self.scalers = {}
        
    def preprocess_text(self, text):
        """
        Preprocesa el texto para la vectorización
        
        Args:
            text (str): Texto a preprocesar
            
        Returns:
            str: Texto preprocesado
        """
        # Tokenizar
        tokens = nltk.word_tokenize(text.lower())
        
        # Eliminar stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Aplicar stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Aplicar lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def create_ngram_representation(self, texts, ngram_range=(1, 1), 
                                   use_tfidf=True, vectorizer_name='unigram'):
        """
        Crea representación basada en n-gramas con TF-IDF o frecuencias
        
        Args:
            texts (list): Lista de textos
            ngram_range (tuple): Rango de n-gramas (min_n, max_n)
            use_tfidf (bool): Usar TF-IDF (True) o frecuencias (False)
            vectorizer_name (str): Nombre para guardar el vectorizador
            
        Returns:
            np.array: Matriz de representaciones vectoriales
        """
        print(f"Creando representación con n-gramas {ngram_range}...")
        
        # Preprocesar textos
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Crear vectorizador
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=self.max_features,
                min_df=2,  # Ignorar términos que aparecen en menos de 2 documentos
                max_df=0.95,  # Ignorar términos que aparecen en más del 95% de documentos
                sublinear_tf=True,  # Aplicar escala logarítmica a TF
                dtype=np.float32
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
                dtype=np.float32
            )
        
        # Ajustar y transformar
        vectors = vectorizer.fit_transform(preprocessed_texts)
        
        # Guardar vectorizador
        self.vectorizers[vectorizer_name] = vectorizer
        
        print(f"  Dimensión del vector: {vectors.shape[1]}")
        print(f"  Vocabulario: {len(vectorizer.vocabulary_)} términos")
        
        return vectors.toarray()
    
    def extract_sentiment_features_vector(self, reviews_with_features):
        """
        Extrae características de sentimiento del Ejercicio 1 y las convierte en vector
        
        Args:
            reviews_with_features (list): Lista de reseñas con características extraídas
            
        Returns:
            np.array: Matriz de características de sentimiento
            list: Nombres de las características
        """
        print("Extrayendo características lingüísticas de sentimiento...")
        
        # Definir qué características numéricas extraer
        numeric_features = [
            # VADER
            'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu',
            
            # SentiWordNet
            'swn_pos_mean', 'swn_neg_mean', 'swn_pos_max', 'swn_neg_max',
            'swn_sentiment_word_count',
            
            # POS tags
            'adj_count', 'adv_count', 'verb_count',
            
            # Negaciones
            'negation_count', 'negated_sentiment_count', 'negative_pattern_count',
            
            # Modificadores
            'intensifier_count', 'diminisher_count', 'modifier_count',
            'intensifier_ratio',
            
            # Dominio
            'domain_mechanics_count', 'domain_components_count',
            'domain_rules_count', 'domain_experience_count',
            'domain_term_count', 'domain_term_ratio',
            'domain_sentiment_count',
            
            # Metadatos
            'token_count', 'sentence_count', 'avg_word_length'
        ]
        
        # Extraer valores para cada reseña
        feature_matrix = []
        for review in reviews_with_features:
            features = review['features']
            feature_vector = []
            
            for feat_name in numeric_features:
                value = features.get(feat_name, 0.0)
                # Asegurar que es numérico
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        
        # Normalizar características (StandardScaler)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Guardar scaler
        self.scalers['sentiment_features'] = scaler
        
        print(f"  Características extraídas: {len(numeric_features)}")
        print(f"  Dimensión del vector: {feature_matrix_scaled.shape[1]}")
        
        # StandardScaler puede devolver float64, lo forzamos de nuevo
        return feature_matrix_scaled.astype(np.float32), numeric_features
    
    def create_combined_representation(self, ngram_vectors, sentiment_vectors):
        """
        Combina representaciones de n-gramas y características de sentimiento
        
        Args:
            ngram_vectors (np.array): Vectores de n-gramas
            sentiment_vectors (np.array): Vectores de características de sentimiento
            
        Returns:
            np.array: Representación combinada
        """
        print("Combinando representaciones...")
        
        # Concatenar horizontalmente
        combined = np.hstack([ngram_vectors, sentiment_vectors])
        
        print(f"  Dimensión combinada: {combined.shape[1]}")
        print(f"    - N-gramas: {ngram_vectors.shape[1]}")
        print(f"    - Sentimiento: {sentiment_vectors.shape[1]}")
        
        return combined


def process_corpus_vectorization(input_file, output_dir, 
                                 ngram_configs=None,
                                 use_stemming=False,
                                 use_lemmatization=True,
                                 remove_stopwords=True,
                                 max_features=5000):
    """
    Procesa el corpus y genera todas las representaciones vectoriales
    
    Args:
        input_file (str): Archivo de entrada con características (del Ejercicio 1)
        output_dir (str): Directorio de salida para guardar representaciones
        ngram_configs (list): Lista de configuraciones de n-gramas
        use_stemming (bool): Aplicar stemming
        use_lemmatization (bool): Aplicar lemmatization
        remove_stopwords (bool): Eliminar stopwords
        max_features (int): Máximo de características para TF-IDF
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuraciones por defecto de n-gramas
    if ngram_configs is None:
        ngram_configs = [
            {'range': (1, 1), 'name': 'unigram', 'use_tfidf': True},
            {'range': (1, 2), 'name': 'unigram_bigram', 'use_tfidf': True},
            {'range': (1, 3), 'name': 'unigram_bigram_trigram', 'use_tfidf': True}
        ]
    
    print("="*60)
    print("GENERACIÓN DE REPRESENTACIONES VECTORIALES")
    print("="*60)
    
    download_resources()

    print(f"\nCargando corpus desde {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"Corpus cargado: {len(corpus)} reseñas")
    
    # Extraer textos
    texts = [review['text'] for review in corpus]
    
    # Inicializar generador de representaciones
    vec_gen = VectorRepresentation(
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization,
        remove_stopwords=remove_stopwords,
        max_features=max_features
    )
    
    # Diccionario para almacenar el RESUMEN (no los datos)
    representation_summary = {}
    
    # 2. REPRESENTACIÓN BASADA EN CARACTERÍSTICAS DE SENTIMIENTO
    # --- Se genera primero porque es pequeña y se reutiliza ---
    print("\n" + "="*60)
    print("2. REPRESENTACIÓN BASADA EN CARACTERÍSTICAS DE SENTIMIENTO")
    print("="*60)
    
    sentiment_vectors, sentiment_feature_names = vec_gen.extract_sentiment_features_vector(corpus)
    
    # Guardar vectores de sentimiento inmediatamente
    output_file_sentiment = os.path.join(output_dir, f'vectors_sentiment.npy')
    np.save(output_file_sentiment, sentiment_vectors)
    print(f"  ✓ Guardado: {output_file_sentiment} (shape: {sentiment_vectors.shape})")
    representation_summary['sentiment'] = {'shape': sentiment_vectors.shape, 'dtype': str(sentiment_vectors.dtype)}

    # 1. & 3. N-GRAMAS Y COMBINADAS (en un solo bucle)
    print("\n" + "="*60)
    print("1. & 3. N-GRAMAS Y REPRESENTACIONES COMBINADAS")
    print("="*60)
    
    for config in ngram_configs:
        ngram_range = config['range']
        name = config['name']
        use_tfidf = config.get('use_tfidf', True)
        
        print(f"\nProcesando: {name}")
        
        # 1. CREAR N-GRAM
        ngram_vectors = vec_gen.create_ngram_representation(
            texts, 
            ngram_range=ngram_range,
            use_tfidf=use_tfidf,
            vectorizer_name=name
        )
        
        # GUARDAR N-GRAM
        output_file_ngram = os.path.join(output_dir, f'vectors_ngram_{name}.npy')
        np.save(output_file_ngram, ngram_vectors)
        print(f"  ✓ Guardado (ngram): {output_file_ngram} (shape: {ngram_vectors.shape})")
        representation_summary[f'ngram_{name}'] = {'shape': ngram_vectors.shape, 'dtype': str(ngram_vectors.dtype)}
    
        # 3. CREAR COMBINADA
        print(f"Combinando n-gramas ({name}) + sentimiento")
        combined_vectors = vec_gen.create_combined_representation(
            ngram_vectors,
            sentiment_vectors
        )
        
        # GUARDAR COMBINADA
        output_file_combined = os.path.join(output_dir, f'vectors_combined_{name}_sentiment.npy')
        np.save(output_file_combined, combined_vectors)
        print(f"  ✓ Guardado (combined): {output_file_combined} (shape: {combined_vectors.shape})")
        representation_summary[f'combined_{name}_sentiment'] = {'shape': combined_vectors.shape, 'dtype': str(combined_vectors.dtype)}

        # En este punto, ngram_vectors y combined_vectors se liberarán de la memoria
        # en la siguiente iteración del bucle.
        
    # 4. GUARDAR OTROS ARCHIVOS (Vectorizadores, Scalers, Metadatos)
    print("\n" + "="*60)
    print("4. GUARDANDO ARCHIVOS COMPLEMENTARIOS")
    print("="*60)
    
    # Guardar vectorizadores y scalers
    vectorizers_file = os.path.join(output_dir, 'vectorizers.pkl')
    with open(vectorizers_file, 'wb') as f:
        pickle.dump(vec_gen.vectorizers, f)
    print(f"  ✓ Guardado: {vectorizers_file}")
    
    scalers_file = os.path.join(output_dir, 'scalers.pkl')
    with open(scalers_file, 'wb') as f:
        pickle.dump(vec_gen.scalers, f)
    print(f"  ✓ Guardado: {scalers_file}")
    
    # Guardar nombres de características de sentimiento
    sentiment_names_file = os.path.join(output_dir, 'sentiment_feature_names.json')
    with open(sentiment_names_file, 'w', encoding='utf-8') as f:
        json.dump(sentiment_feature_names, f, indent=2)
    print(f"  ✓ Guardado: {sentiment_names_file}")
    
    # Guardar ratings para referencia
    metadata = {
        'game_ids': [review.get('game_id') for review in corpus],
        'ratings': [review.get('rating') for review in corpus],
        'num_reviews': len(corpus)
    }
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Guardado: {metadata_file}")
    
    # Guardar configuración utilizada
    config_info = {
        'use_stemming': use_stemming,
        'use_lemmatization': use_lemmatization,
        'remove_stopwords': remove_stopwords,
        'max_features': max_features,
        'ngram_configs': ngram_configs,
        'representations': representation_summary # Usar el resumen
    }
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2)
    print(f"  ✓ Guardado: {config_file}")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE REPRESENTACIONES GENERADAS")
    print("="*60)
    
    print(f"\nTotal de representaciones: {len(representation_summary)}")
    print("\nDetalle:")
    # Usar el resumen para mostrar los detalles
    for name, info in sorted(representation_summary.items()):
        print(f"  • {name:40s} → {info['shape']}")
    
    print(f"\n✓ Todos los archivos guardados en: {output_dir}")
    print("\nArchivos generados:")
    print(f"  • {len(representation_summary)} archivos .npy con vectores")
    print(f"  • vectorizers.pkl (vectorizadores TF-IDF/Count)")
    print(f"  • scalers.pkl (normalizadores)")
    print(f"  • sentiment_feature_names.json (nombres de características)")
    print(f"  • metadata.json (IDs y ratings)")
    print(f"  • config.json (configuración utilizada)")
    
    print("\n" + "="*60)
    print("¡VECTORIZACIÓN COMPLETADA!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Genera representaciones vectoriales de reseñas'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='corpus_con_features.json',
        help='Archivo de entrada con características (del Ejercicio 1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='vectores',
        help='Directorio de salida para las representaciones'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Número máximo de características para TF-IDF'
    )
    parser.add_argument(
        '--no-lemmatization',
        action='store_true',
        help='Desactivar lemmatization'
    )
    parser.add_argument(
        '--use-stemming',
        action='store_true',
        help='Activar stemming'
    )
    parser.add_argument(
        '--keep-stopwords',
        action='store_true',
        help='No eliminar stopwords'
    )
    
    args = parser.parse_args()
    
    process_corpus_vectorization(
        input_file=args.input,
        output_dir=args.output_dir,
        use_stemming=args.use_stemming,
        use_lemmatization=not args.no_lemmatization,
        remove_stopwords=not args.keep_stopwords,
        max_features=args.max_features
    )
