"""
Práctica 2 - Ejercicio 1: Extracción de características lingüísticas
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid
"""

import json
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from collections import Counter
import re

# Descargar recursos necesarios (ejecutar solo la primera vez)
def download_resources():
    """Descarga los recursos necesarios de NLTK"""
    resources = [
        'sentiwordnet',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'punkt',
        'punkt_tab',
        'omw-1.4'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            print(f"No se pudo descargar: {resource}")


class ManualSentimentAnalyzer:
    """Analizador de sentimiento manual basado en lexicones"""
    
    def __init__(self):
        """Inicializa el analizador con lexicones de sentimiento"""
        # Palabras positivas comunes en reseñas de juegos
        self.positive_words = {
            'excellent', 'amazing', 'great', 'awesome', 'fantastic', 'wonderful',
            'outstanding', 'superb', 'brilliant', 'perfect', 'best', 'love',
            'enjoy', 'fun', 'engaging', 'exciting', 'interesting', 'creative',
            'innovative', 'beautiful', 'gorgeous', 'stunning', 'impressive',
            'addictive', 'masterpiece', 'recommend', 'worth', 'solid', 'good',
            'nice', 'pleasant', 'entertaining', 'satisfying', 'rewarding',
            'clever', 'unique', 'immersive', 'challenging', 'strategic',
            'tactical', 'deep', 'rich', 'polished', 'smooth', 'elegant',
            'accessible', 'balanced', 'replayable', 'thematic', 'quality',
            'premium', 'detailed', 'well-designed', 'works', 'liked', 'favorite'
        }
        
        # Palabras negativas comunes en reseñas de juegos
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'boring', 'dull',
            'tedious', 'frustrating', 'disappointing', 'broken', 'unbalanced',
            'unfair', 'overpriced', 'expensive', 'waste', 'poor', 'cheap',
            'ugly', 'confusing', 'complicated', 'unclear', 'messy', 'clunky',
            'slow', 'dragging', 'repetitive', 'monotonous', 'luck-based',
            'random', 'chaotic', 'unplayable', 'hate', 'dislike', 'avoid',
            'lacks', 'missing', 'limited', 'shallow', 'simple', 'bland',
            'generic', 'derivative', 'forgettable', 'mediocre', 'overrated',
            'downtime', 'fiddly', 'dated', 'outdated', 'flawed', 'issues',
            'problems', 'fails', 'difficult', 'hard', 'punishing', 'useless'
        }
        
        # Intensificadores (aumentan el sentimiento)
        self.intensifiers = {
            'very', 'extremely', 'absolutely', 'completely', 'totally',
            'highly', 'incredibly', 'exceptionally', 'particularly',
            'really', 'super', 'way', 'utterly', 'quite', 'greatly',
            'especially', 'truly', 'genuinely', 'definitely', 'certainly'
        }
        
        # Palabras de negación
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'hardly', 'scarcely', 'barely', "n't", 'none',
            'without', 'lack', 'lacks'
        }
    
    def calculate_sentiment_score(self, word, pos_tag=None):
        """
        Calcula el score de sentimiento de una palabra
        
        Args:
            word (str): Palabra a analizar
            pos_tag (str): POS tag opcional
            
        Returns:
            float: Score entre -1 (negativo) y 1 (positivo)
        """
        word_lower = word.lower()
        
        # Buscar en lexicones manuales
        if word_lower in self.positive_words:
            return 1.0
        elif word_lower in self.negative_words:
            return -1.0
        
        # Si no está en los lexicones básicos, intentar con SentiWordNet
        return 0.0
    
    def polarity_scores(self, text):
        """
        Calcula scores de polaridad para un texto (simula VADER)
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            dict: Diccionario con scores pos, neg, neu, compound
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Detectar negaciones y calcular scores
        negation_window = False
        negation_countdown = 0
        
        for i, (token, tag) in enumerate(pos_tags):
            # Verificar si es negación
            if token in self.negation_words:
                negation_window = True
                negation_countdown = 3  # Ventana de 3 palabras
                continue
            
            # Calcular score base de la palabra
            base_score = self.calculate_sentiment_score(token, tag)
            
            # Buscar también en SentiWordNet
            if base_score == 0.0:
                swn_pos, swn_neg, _ = self.get_sentiwordnet_score(token, tag)
                if swn_pos > swn_neg:
                    base_score = swn_pos
                elif swn_neg > swn_pos:
                    base_score = -swn_neg
            
            # Aplicar modificadores
            modified_score = base_score
            
            # Verificar intensificador antes de la palabra
            if i > 0 and pos_tags[i-1][0].lower() in self.intensifiers:
                modified_score *= 1.5
            
            # Aplicar negación si estamos en ventana de negación
            if negation_window and negation_countdown > 0:
                modified_score *= -0.8
                negation_countdown -= 1
                if negation_countdown == 0:
                    negation_window = False
            
            # Contar por tipo
            if modified_score > 0:
                positive_count += 1
                scores.append(modified_score)
            elif modified_score < 0:
                negative_count += 1
                scores.append(modified_score)
            else:
                neutral_count += 1
        
        # Calcular proporciones
        total_words = len(tokens)
        if total_words == 0:
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
        
        pos_proportion = positive_count / total_words
        neg_proportion = negative_count / total_words
        neu_proportion = neutral_count / total_words
        
        # Calcular compound score (normalizado entre -1 y 1)
        if scores:
            compound = sum(scores) / len(scores)
            # Normalizar usando función sigmoide-like
            compound = compound / (abs(compound) + 1)
        else:
            compound = 0.0
        
        return {
            'pos': pos_proportion,
            'neg': neg_proportion,
            'neu': neu_proportion,
            'compound': compound
        }
    
    def get_sentiwordnet_score(self, word, pos_tag):
        """
        Obtiene score de SentiWordNet
        
        Args:
            word (str): Palabra
            pos_tag (str): POS tag
            
        Returns:
            tuple: (pos_score, neg_score, obj_score)
        """
        wn_pos = self.get_wordnet_pos(pos_tag)
        if wn_pos is None:
            return 0.0, 0.0, 0.0
        
        try:
            synsets = list(swn.senti_synsets(word, wn_pos))
            if synsets:
                synset = synsets[0]
                return synset.pos_score(), synset.neg_score(), synset.obj_score()
        except:
            pass
        
        return 0.0, 0.0, 0.0
    
    def get_wordnet_pos(self, treebank_tag):
        """Convierte POS tags de Penn Treebank a WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None


class FeatureExtractor:
    """Extractor de características lingüísticas para análisis de sentimiento"""
    
    def __init__(self):
        """Inicializa el extractor con los recursos necesarios"""
        # Inicializar analizador de sentimiento manual
        self.manual_sentiment = ManualSentimentAnalyzer()
        
        # Palabras de negación
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
            'nowhere', 'hardly', 'scarcely', 'barely', "n't", 'none',
            'fail', 'lacks', 'without', 'lack'
        }
        
        # Intensificadores (amplificadores)
        self.intensifiers = {
            'very', 'extremely', 'absolutely', 'completely', 'totally',
            'highly', 'incredibly', 'exceptionally', 'particularly',
            'really', 'super', 'way', 'utterly', 'quite', 'greatly'
        }
        
        # Atenuadores (mitigadores)
        self.diminishers = {
            'slightly', 'somewhat', 'barely', 'hardly', 'scarcely',
            'sort of', 'kind of', 'a bit', 'a little', 'fairly',
            'rather', 'pretty', 'relatively', 'moderately'
        }
        
        # Vocabulario de dominio de juegos de mesa
        self.domain_vocabulary = {
            # Mecánicas de juego
            'mechanics': ['dice', 'roll', 'card', 'deck', 'strategy', 'tactical',
                         'combat', 'resource', 'worker', 'placement', 'drafting',
                         'engine', 'building', 'cooperative', 'competitive'],
            
            # Componentes
            'components': ['miniature', 'miniatures', 'token', 'tokens', 'board',
                          'artwork', 'quality', 'piece', 'pieces', 'component',
                          'components', 'meeple', 'meeples'],
            
            # Reglas y jugabilidad
            'rules': ['rule', 'rules', 'rulebook', 'setup', 'turn', 'round',
                     'phase', 'action', 'gameplay', 'playtime', 'duration',
                     'complexity', 'learning'],
            
            # Experiencia de juego
            'experience': ['fun', 'boring', 'engaging', 'replayability', 'replay',
                          'theme', 'thematic', 'immersive', 'interactive',
                          'downtime', 'player', 'players', 'scaling']
        }
        
        # Crear conjunto completo de términos de dominio
        self.all_domain_terms = set()
        for category in self.domain_vocabulary.values():
            self.all_domain_terms.update(category)
    
    def get_wordnet_pos(self, treebank_tag):
        """Convierte POS tags de Penn Treebank a WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    def get_sentiwordnet_score(self, word, pos_tag):
        """Obtiene el score de SentiWordNet para una palabra"""
        wn_pos = self.get_wordnet_pos(pos_tag)
        if wn_pos is None:
            return 0.0, 0.0, 0.0
        
        try:
            synsets = list(swn.senti_synsets(word, wn_pos))
            if synsets:
                # Usar el primer synset (más común)
                synset = synsets[0]
                return synset.pos_score(), synset.neg_score(), synset.obj_score()
        except:
            pass
        
        return 0.0, 0.0, 0.0
    
    def extract_sentiment_words(self, text, tokens, pos_tags):
        """
        Extrae palabras de opinión/sentimiento usando analizador manual y SentiWordNet
        
        Returns:
            dict: Características de palabras de sentimiento
        """
        features = {}
        
        # 1. Análisis manual de sentimiento (reemplaza VADER)
        manual_scores = self.manual_sentiment.polarity_scores(text)
        features['vader_compound'] = manual_scores['compound']
        features['vader_pos'] = manual_scores['pos']
        features['vader_neg'] = manual_scores['neg']
        features['vader_neu'] = manual_scores['neu']
        
        # 2. Análisis con SentiWordNet (nivel de palabra)
        pos_scores = []
        neg_scores = []
        obj_scores = []
        
        sentiment_words = []
        
        for token, pos in pos_tags:
            if len(token) > 2:  # Ignorar palabras muy cortas
                pos_score, neg_score, obj_score = self.get_sentiwordnet_score(token.lower(), pos)
                
                if pos_score > 0 or neg_score > 0:
                    pos_scores.append(pos_score)
                    neg_scores.append(neg_score)
                    obj_scores.append(obj_score)
                    
                    # Guardar palabras con sentimiento significativo
                    if pos_score > 0.3 or neg_score > 0.3:
                        sentiment_words.append({
                            'word': token,
                            'pos': pos,
                            'pos_score': pos_score,
                            'neg_score': neg_score
                        })
        
        # Estadísticas agregadas de SentiWordNet
        features['swn_pos_mean'] = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        features['swn_neg_mean'] = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
        features['swn_pos_max'] = max(pos_scores) if pos_scores else 0.0
        features['swn_neg_max'] = max(neg_scores) if neg_scores else 0.0
        features['swn_sentiment_word_count'] = len(sentiment_words)
        
        # Contar adjetivos, adverbios y verbos (POS tags relevantes para sentimiento)
        features['adj_count'] = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
        features['adv_count'] = sum(1 for _, pos in pos_tags if pos.startswith('RB'))
        features['verb_count'] = sum(1 for _, pos in pos_tags if pos.startswith('VB'))
        
        features['sentiment_words'] = sentiment_words
        
        return features
    
    def extract_negations(self, text, tokens, pos_tags):
        """
        Detecta patrones de negación en el texto
        
        Returns:
            dict: Características de negación
        """
        features = {}
        tokens_lower = [t.lower() for t in tokens]
        
        # Contar palabras de negación
        negation_count = sum(1 for token in tokens_lower if token in self.negation_words)
        features['negation_count'] = negation_count
        
        # Detectar negaciones contextuales (ventana de 3 palabras)
        negated_sentiments = []
        for i, token in enumerate(tokens_lower):
            if token in self.negation_words:
                # Buscar palabras de sentimiento en las siguientes 3 palabras
                window = tokens[i+1:min(i+4, len(tokens))]
                for j, word in enumerate(window):
                    pos_score, neg_score, _ = self.get_sentiwordnet_score(
                        word.lower(), 
                        pos_tags[i+1+j][1] if i+1+j < len(pos_tags) else 'NN'
                    )
                    if pos_score > 0.3 or neg_score > 0.3:
                        negated_sentiments.append({
                            'negation': token,
                            'target': word,
                            'original_pos': pos_score,
                            'original_neg': neg_score
                        })
        
        features['negated_sentiment_count'] = len(negated_sentiments)
        features['negated_sentiments'] = negated_sentiments
        
        # Detectar frases negativas comunes
        text_lower = text.lower()
        negative_patterns = [
            'not good', 'not great', 'not fun', 'not worth',
            'fail to', 'lacks', 'without any', 'hardly any'
        ]
        features['negative_pattern_count'] = sum(
            1 for pattern in negative_patterns if pattern in text_lower
        )
        
        return features
    
    def extract_modifiers(self, tokens):
        """
        Extrae intensificadores y atenuadores
        
        Returns:
            dict: Características de modificadores
        """
        features = {}
        tokens_lower = [t.lower() for t in tokens]
        
        # Contar intensificadores
        intensifier_count = sum(1 for token in tokens_lower if token in self.intensifiers)
        features['intensifier_count'] = intensifier_count
        
        # Contar atenuadores
        diminisher_count = sum(1 for token in tokens_lower if token in self.diminishers)
        features['diminisher_count'] = diminisher_count
        
        # Ratio de modificadores
        total_modifiers = intensifier_count + diminisher_count
        features['modifier_count'] = total_modifiers
        features['intensifier_ratio'] = (
            intensifier_count / total_modifiers if total_modifiers > 0 else 0.0
        )
        
        # Detectar modificadores específicos usados
        intensifiers_found = [t for t in tokens_lower if t in self.intensifiers]
        diminishers_found = [t for t in tokens_lower if t in self.diminishers]
        
        features['intensifiers_found'] = intensifiers_found
        features['diminishers_found'] = diminishers_found
        
        return features
    
    def extract_domain_vocabulary(self, text, tokens):
        """
        Extrae características relacionadas con vocabulario de dominio
        
        Returns:
            dict: Características de vocabulario de dominio
        """
        features = {}
        tokens_lower = [t.lower() for t in tokens]
        text_lower = text.lower()
        
        # Contar menciones por categoría
        for category, terms in self.domain_vocabulary.items():
            count = sum(1 for token in tokens_lower if token in terms)
            features[f'domain_{category}_count'] = count
        
        # Contar total de términos de dominio
        domain_term_count = sum(1 for token in tokens_lower if token in self.all_domain_terms)
        features['domain_term_count'] = domain_term_count
        features['domain_term_ratio'] = (
            domain_term_count / len(tokens) if tokens else 0.0
        )
        
        # Detectar co-ocurrencia de términos de dominio con sentimiento
        domain_sentiment_collocations = []
        for i, token in enumerate(tokens_lower):
            if token in self.all_domain_terms:
                # Buscar palabras de sentimiento en ventana de ±3 palabras
                window_start = max(0, i-3)
                window_end = min(len(tokens), i+4)
                window = tokens_lower[window_start:window_end]
                
                # Verificar si hay palabras de sentimiento en la ventana usando analizador manual
                vader_window_text = ' '.join(window)
                manual_score = self.manual_sentiment.polarity_scores(vader_window_text)
                
                if abs(manual_score['compound']) > 0.3:
                    domain_sentiment_collocations.append({
                        'domain_term': token,
                        'context': ' '.join(tokens[window_start:window_end]),
                        'sentiment': manual_score['compound']
                    })
        
        features['domain_sentiment_collocations'] = domain_sentiment_collocations
        features['domain_sentiment_count'] = len(domain_sentiment_collocations)
        
        return features
    
    def extract_all_features(self, review_text):
        """
        Extrae todas las características lingüísticas de una reseña
        
        Args:
            review_text (str): Texto de la reseña
            
        Returns:
            dict: Diccionario con todas las características extraídas
        """
        # Tokenización y POS tagging
        tokens = word_tokenize(review_text)
        pos_tags = pos_tag(tokens)
        
        # Extraer todas las características
        features = {}
        
        # 1. Palabras de sentimiento
        sentiment_features = self.extract_sentiment_words(review_text, tokens, pos_tags)
        features.update(sentiment_features)
        
        # 2. Negaciones
        negation_features = self.extract_negations(review_text, tokens, pos_tags)
        features.update(negation_features)
        
        # 3. Modificadores
        modifier_features = self.extract_modifiers(tokens)
        features.update(modifier_features)
        
        # 4. Vocabulario de dominio
        domain_features = self.extract_domain_vocabulary(review_text, tokens)
        features.update(domain_features)
        
        # Agregar metadatos básicos
        features['token_count'] = len(tokens)
        features['sentence_count'] = len([c for c in review_text if c in '.!?'])
        features['avg_word_length'] = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        
        return features


def process_corpus(corpus_file, output_file):
    """
    Procesa un corpus de reseñas y extrae características
    
    Args:
        corpus_file (str): Ruta al archivo del corpus (JSON)
        output_file (str): Ruta al archivo de salida con características
    """
    print("Descargando recursos de NLTK...")
    download_resources()
    
    print("Cargando corpus...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print("Inicializando extractor de características...")
    extractor = FeatureExtractor()
    
    print(f"Procesando reseñas...")
    processed_reviews = []
    
    for game_id, game_data in corpus["games"].items():
        for i, review in enumerate(game_data["comments"]):
            text = review.get("comment", "")
            rating = review.get("rating", None)
            features = extractor.extract_all_features(text)
            
            processed_review = {
                'game_id': game_id,
                'text': text,
                'rating': rating,
                'features': features
            }
            processed_reviews.append(processed_review)

    
    print(f"Guardando resultados en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_reviews, f, indent=2, ensure_ascii=False)
    
    print("¡Extracción de características completada!")
    print(f"Total de reseñas procesadas: {len(processed_reviews)}")
    
    # Mostrar estadísticas
    print("\n--- Estadísticas de características extraídas ---")
    avg_features = {}
    for review in processed_reviews:
        for key, value in review['features'].items():
            if isinstance(value, (int, float)):
                if key not in avg_features:
                    avg_features[key] = []
                avg_features[key].append(value)
    
    print("\nPromedios de características numéricas:")
    for key, values in sorted(avg_features.items()):
        print(f"  {key}: {sum(values)/len(values):.4f}")


if __name__ == "__main__":
    # Ejemplo de uso
    # Ajustar las rutas según la estructura de archivos
    CORPUS_INPUT = "bgg_short.json"  # Corpus de la Práctica 1
    CORPUS_OUTPUT = "corpus_con_features.json"  # Corpus con características
    
    process_corpus(CORPUS_INPUT, CORPUS_OUTPUT)