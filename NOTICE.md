# Licence Scope and Research Provenance

The new source code and original documentation in this repository are available under the root MIT License.

This 2026 implementation is a rewrite of the architecture described in Junjia Cao's IEEE ISPCEM 2024 paper, "An improved RCBT model and its application to sentiment analysis of movie comments" (DOI: 10.1109/ISPCEM64498.2024.00180). The IEEE publication is not redistributed here.

The historical 2024 experiment tree extended https://github.com/BeiCunNan/sentiment_analysis_Imdb, which declared no licence when reviewed. That historical source tree, its dataset CSV, and its generated figures are not included or redistributed in this repository. The new model, data pipeline, and training workflow are independently structured and preserve sequence length throughout the classifier.

Historical paper metrics are not presented as results of this implementation. Any verification results in the README describe only the commands and scope explicitly stated there.

IMDB data are loaded through the Hugging Face `imdb` dataset and remain subject to the original dataset terms. Pretrained encoders are loaded from Hugging Face and retain their respective licences.
