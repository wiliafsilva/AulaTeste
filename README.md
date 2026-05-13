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
