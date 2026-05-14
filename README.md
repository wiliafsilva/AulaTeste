# AulaTeste

Este repositório contém exemplos de execução em Python usados na aula.

## Requisitos

- Python 3.8–3.11 (recomendado: 3.11)
- `pip` instalado

## Passo a passo de execução

1. Abra um terminal e navegue até a pasta do projeto:

	 - PowerShell:

		 ```powershell
		 cd "C:\Users\wilia.silva\Desktop\Nova pasta\AulaTeste"
		 ```

2. (Opcional) Crie e ative um ambiente virtual:

	 - PowerShell:

		 ```powershell
		 python -m venv venv
		 .\venv\Scripts\Activate.ps1
		 ```

	 - Prompt de Comando (cmd):

		 ```cmd
		 python -m venv venv
		 venv\Scripts\activate.bat
		 ```

3. Instale as dependências (se houver):

	 ```Atualizar requirements
	 pip freeze > requirements.txt
	 ```
	 
	 ```bash
	 pip install -r requirements.txt
	 ```

4. Execute os exemplos:

	 - Executar o script simples:

		 ```bash
		 python simples.py
		 ```

	 - Executar o script completo:

		 ```bash
		 python completo.py
		 ```

5. Para desativar o ambiente virtual:

	 ```bash
	 deactivate
	 ```

## Para instalar modo completo:

3. Crie venv usando Python 3.11

Verifique:

py -3.11 --version

Depois:

py -3.11 -m venv venv

Ative:

venv\Scripts\activate
4. Instale stack compatível

Atualize pip:

python -m pip install --upgrade pip

Agora instale ESTE conjunto exato:

pip install numpy==1.23.5
pip install scipy==1.10.1
pip install pandas==2.0.3
pip install scikit-learn==1.3.2
pip install scikit-image==0.21.0
pip install pillow==10.4.0
pip install opencv-python
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install py-feat==0.6.1

Instale:

pip install fer

pip install tensorflow