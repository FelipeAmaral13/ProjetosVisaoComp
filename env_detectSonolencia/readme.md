# Webcam App

Este é um aplicativo simples de webcam que utiliza as bibliotecas OpenCV e dlib para realizar a detecção dos pontos de referência faciais e calcular a Razão de Aspecto do Olho (EAR) em tempo real. O aplicativo exibe o vídeo da webcam com o valor do EAR calculado e emite um alerta caso seja detectado sono.

![Vídeo sem título ‐ Feito com o Clipchamp](https://github.com/FelipeAmaral13/ProjetosVisaoComp/assets/5797933/0d9c1461-9a25-4f87-9848-24a5af08342e)


## Pré-requisitos

Antes de executar o aplicativo, certifique-se de ter as seguintes dependências instaladas:

- Python 3.x
- OpenCV
- dlib
- numpy
- Pillow
- imutils
- scipy

Você pode instalar as dependências necessárias executando o seguinte comando:

```
pip install opencv-python dlib numpy Pillow imutils scipy
```

Além disso, faça o download do arquivo `shape_predictor_68_face_landmarks.dat` e coloque-o no diretório `data`, localizado no mesmo diretório do seu código. Esse arquivo é necessário para a detecção dos pontos de referência faciais.

## Uso

Para iniciar o aplicativo de webcam, execute o seguinte comando:

```
python webcam_app.py
```

Assim que o aplicativo iniciar, você verá uma interface gráfica com um botão "Iniciar". Clicar no botão "Iniciar" iniciará a captura de vídeo da webcam. O aplicativo irá continuamente detectar os pontos de referência faciais, calcular o EAR e exibir o vídeo com o valor correspondente do EAR. Se o valor do EAR calculado ficar abaixo de um determinado limite, indicando sonolência, uma mensagem de alerta será exibida.

Você pode parar a captura da webcam clicando no botão "Parar". O aplicativo pode ser fechado ao fechar a janela do aplicativo.

## Referências

Este projeto foi desenvolvido com base nas seguintes referências:

- Documentação do OpenCV: https://docs.opencv.org/
- Documentação do dlib: http://dlib.net/
- Blog PyImageSearch por Adrian Rosebrock: https://www.pyimagesearch.com/
- Livro: "Python for Computer Vision with OpenCV and Deep Learning" por Adrian Rosebrock: https://www.pyimagesearch.com/pyimagesearch-gurus/

Consulte esses recursos para obter explicações mais detalhadas e tutoriais sobre detecção de pontos de referência faciais, cálculo da razão de aspecto do olho e técnicas de visão computacional.

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE](LICENSE) para obter mais detalhes.
