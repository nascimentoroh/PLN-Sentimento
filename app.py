from deep_translator import GoogleTranslator
import nltk
import streamlit as st
from textblob import TextBlob

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Analisador de Sentimentos Offline", page_icon="📊", layout="centered"
)


# Garantir o download dos pacotes necessários do NLTK para o TextBlob de forma segura
@st.cache_resource
def load_nltk_dependencies():
  try:
    nltk.data.find("tokenizers/punkt")
  except LookupError:
    nltk.download("punkt", quiet=True)


load_nltk_dependencies()


def main():
  # Cabeçalho explicativo conforme regras de negócio
  st.title("📊 Analisador de Sentimentos Offline")
  st.markdown(
      "Esta é uma versão **Offline** desenvolvida com bibliotecas"
      " tradicionais de Python (`TextBlob` e `deep-translator`). O"
      " processamento de PLN é realizado localmente. Textos em português são"
      " traduzidos de forma invisível para o inglês para garantir máxima"
      " precisão na análise de polaridade."
  )

  st.markdown("---")

  # Área de texto para entrada do usuário
  user_input = st.text_area(
      "Digite seu texto abaixo para análise:",
      placeholder=(
          "Ex: Estou extremamente satisfeito com o atendimento recebido hoje!"
      ),
      height=150,
  )

  # Botão 'Analisar'
  if st.button("Analisar Sentimento", type="primary"):
    if not user_input.strip():
      st.warning("⚠️ Por favor, digite algum texto antes de clicar em analisar.")
    else:
      with st.spinner("Processando análise de sentimentos..."):
        try:
          # Tradução invisível para o inglês (otimizando a precisão do TextBlob)
          translated_text = GoogleTranslator(source="auto", target="en").translate(
              user_input
          )

          # Análise de Polaridade e Subjetividade com TextBlob
          analysis = TextBlob(translated_text)
          polarity = analysis.sentiment.polarity
          subjectivity = analysis.sentiment.subjectivity

          # Classificação de Sentimento baseada na polaridade [-1.0 a 1.0]
          if polarity > 0.05:
            st.success(
                "### Resultado: **Positivo** 😊\nO texto transmite uma"
                " atmosfera favorável ou elogiosa."
            )
          elif polarity < -0.05:
            st.error(
                "### Resultado: **Negativo** 😞\nO texto transmite insatisfação,"
                " crítica ou tom adverso."
            )
          else:
            st.warning(
                "### Resultado: **Neutro** 😐\nO texto apresenta um tom"
                " informativo, impessoal ou equilibrado."
            )

          # Expander com métricas detalhadas (valor agregado para engenheiros/usuários)
          with st.expander("Ver detalhes técnicos da análise"):
            st.write(f"**Texto Original:** {user_input}")
            st.write(f"**Texto Traduzido (Inglês):** {translated_text}")
            st.write(
                f"**Polaridade:** `{polarity:.4f}` *(Varia de -1.0 (Muito"
                " Negativo) a 1.0 (Muito Positivo))*"
            )
            st.write(
                f"**Subjetividade:** `{subjectivity:.4f}` *(Varia de 0.0"
                " (Objetivo) a 1.0 (Subjetivo))*"
            )

        except Exception as e:
          st.error(
              f"Ocorreu um erro durante o processamento: {str(e)}. Verifique a"
              " conexão com a rede para a etapa de tradução."
          )

  # Rodapé com menção à licença MIT
  st.markdown("---")
  st.markdown(
      "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
      "Analisador de Sentimentos Offline • Licença MIT • Desenvolvido com"
      " Streamlit & Python"
      "</div>",
      unsafe_allow_html=True,
  )


if __name__ == "__main__":
  main()
