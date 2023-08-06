# Classificação de Tumores no Cérebro com Vision Transformers

## Introdução

Este repositório contém o código em PyTorch para treinar e utilizar um modelo de detecção de tumores cerebrais a partir de imagens de ressonância magnética (MRI). O modelo utiliza a arquitetura Vision Transformers (ViT), uma abordagem de aprendizado profundo que se mostrou muito eficaz em tarefas de visão computacional, incluindo classificação de imagens.

## Sobre o Modelo

O modelo ViTFinetune é implementado como uma classe em PyTorch e pode ser facilmente personalizado para outras tarefas de classificação de imagem. Ele utiliza um modelo pré-treinado do ViT da biblioteca Hugging Face Transformers e realiza o fine-tuning para a tarefa específica de detecção de tumores cerebrais.

## Sobre o Conjunto de Dados

O conjunto de dados utilizado neste projeto contém três pastas: "yes," "no," e "pred", com um total de 3060 imagens de ressonância magnética do cérebro. A pasta "yes" contém 1500 imagens com tumores cerebrais, representando a classe positiva para a detecção de tumores cerebrais, enquanto a pasta "no" contém 1500 imagens sem tumores cerebrais, representando a classe negativa. O conjunto de dados pode ser baixado neste link: [Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

## Instruções de Uso

1. Baixe o conjunto de dados do link fornecido acima e organize as imagens nas pastas "yes" e "no" de acordo com suas respectivas classes.
2. Clone este repositório em seu ambiente local.
3. Instale as dependências necessárias especificadas no arquivo `requirements.txt`.
4. Utilize o arquivo python fornecido para treinar o modelo ViTFinetune com os dados fornecidos.
5. Após o treinamento, utilize o método `predict_image(image_path)` para fazer inferências em imagens individuais.

## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões de melhorias ou encontrar problemas, por favor, abra uma "issue" no repositório para discutirmos.

## Referências

- [Vision Transformers (ViT) - Hugging Face Transformers](https://huggingface.co/transformers/model_doc/vit.html)
- [Dataset - Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
