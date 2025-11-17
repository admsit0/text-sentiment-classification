# Informe de Resultados - Práctica 2
## Clasificación de Texto para Análisis de Sentimiento

**Fecha:** 2025-11-17

---

## 1. Configuración del Experimento

### Etiquetado de Clases

Umbrales de rating utilizados:

- **Positiva**: Rating ≥ 7
- **Neutra**: 5 ≤ Rating ≤ 6
- **Negativa**: Rating ≤ 4

Distribución de clases (antes del balanceo):

- Positivas: 55673
- Neutras: 25193
- Negativas: 9436

### Representaciones Vectoriales

Total de representaciones evaluadas: 7

- combined_unigram_bigram_sentiment
- combined_unigram_bigram_trigram_sentiment
- combined_unigram_sentiment
- ngram_unigram
- ngram_unigram_bigram
- ngram_unigram_bigram_trigram
- sentiment

### Modelos de Clasificación

Total de modelos evaluados: 4

- Multinomial Naive Bayes
- Random Forest
- SVM (LinearSVC)
- SVM (RBF)

---

## 2. Mejor Modelo

**Modelo:** Multinomial Naive Bayes

**Representación:** combined_unigram_sentiment

### Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.5380 |
| Precision (macro) | 0.5356 |
| Recall (macro) | 0.5380 |
| F1-score (macro) | 0.5361 |
| CV Score | 0.5271 |
| Val Score | 0.5203 |

---

## 3. Top 10 Modelos

| Rank | Modelo | Representación | F1-score | Accuracy |
|------|--------|----------------|----------|----------|
| 1 | Multinomial Naive Bayes | combined_unigram_sentiment | 0.5361 | 0.5380 |
| 2 | Multinomial Naive Bayes | ngram_unigram_bigram | 0.5330 | 0.5359 |
| 3 | Multinomial Naive Bayes | combined_unigram_bigram_sentiment | 0.5326 | 0.5362 |
| 4 | Multinomial Naive Bayes | ngram_unigram | 0.5322 | 0.5332 |
| 5 | Multinomial Naive Bayes | combined_unigram_bigram_trigram_sentiment | 0.5322 | 0.5360 |
| 6 | Multinomial Naive Bayes | ngram_unigram_bigram_trigram | 0.5316 | 0.5348 |
| 7 | Random Forest | ngram_unigram_bigram | 0.5295 | 0.5313 |
| 8 | Random Forest | ngram_unigram_bigram_trigram | 0.5286 | 0.5298 |
| 9 | Random Forest | combined_unigram_bigram_sentiment | 0.5235 | 0.5341 |
| 10 | Random Forest | combined_unigram_bigram_trigram_sentiment | 0.5226 | 0.5336 |

---

## 4. Análisis de Representaciones

| Representación                            |   F1 Promedio |   F1 Std |   F1 Máximo |
|:------------------------------------------|--------------:|---------:|------------:|
| combined_unigram_bigram_sentiment         |        0.5135 |   0.0179 |      0.5326 |
| combined_unigram_bigram_trigram_sentiment |        0.5129 |   0.0177 |      0.5322 |
| ngram_unigram_bigram                      |        0.5079 |   0.0274 |      0.533  |
| ngram_unigram_bigram_trigram              |        0.507  |   0.027  |      0.5316 |
| combined_unigram_sentiment                |        0.5035 |   0.0308 |      0.5361 |
| ngram_unigram                             |        0.5025 |   0.0292 |      0.5322 |
| sentiment                                 |        0.4486 |   0.0288 |      0.4754 |

---

## 5. Análisis de Modelos

| Modelo                  |   F1 Promedio |   F1 Std |   F1 Máximo |
|:------------------------|--------------:|---------:|------------:|
| Random Forest           |        0.5163 |   0.0188 |      0.5295 |
| Multinomial Naive Bayes |        0.5154 |   0.0463 |      0.5361 |
| SVM (LinearSVC)         |        0.4885 |   0.0214 |      0.5049 |
| SVM (RBF)               |        0.4774 |   0.0124 |      0.4929 |

---

## 6. Conclusiones

- El mejor modelo obtenido fue **Multinomial Naive Bayes** con la representación **combined_unigram_sentiment**, alcanzando un F1-score (macro) de **0.5361**.

- La representación que mejor funcionó en promedio fue **combined_unigram_bigram_sentiment** con un F1-score promedio de **0.5135**.

- El tipo de modelo más efectivo fue **Random Forest** con un F1-score promedio de **0.5163**.

