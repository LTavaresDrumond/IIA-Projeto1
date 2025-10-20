# Configuração da IGDB API para Imagens de Jogos

## Como obter as credenciais da IGDB API

1. **Acesse o Console de Desenvolvedores da Twitch**
   - Vá para: https://dev.twitch.tv/console
   - Faça login com sua conta Twitch (ou crie uma se não tiver)

2. **Registre sua Aplicação**
   - Clique em "Register Your Application"
   - Preencha os campos:
     - **Name**: NextGame App (ou qualquer nome)
     - **OAuth Redirect URLs**: http://localhost
     - **Category**: Application Integration
   - Clique em "Create"

3. **Obtenha suas Credenciais**
   - Após criar, você verá seu **Client ID**
   - Clique em "New Secret" para gerar um **Client Secret**
   - Copie ambos (Client ID e Client Secret)

## Configuração no Streamlit

### Opção 1: Usando st.secrets (Recomendado para produção)

Crie um arquivo `.streamlit/secrets.toml` na raiz do projeto:

```toml
IGDB_CLIENT_ID = "seu_client_id_aqui"
IGDB_CLIENT_SECRET = "seu_client_secret_aqui"
```

### Opção 2: Modo de Desenvolvimento (Placeholders)

Se não configurar as credenciais, o sistema usará placeholders automáticos com o nome do jogo.

## Testando

Após configurar:
1. Reinicie o aplicativo Streamlit
2. As imagens dos jogos serão carregadas automaticamente da IGDB
3. O sistema usa cache para evitar requisições repetidas

## Notas Importantes

- ✅ **Cache**: As imagens são armazenadas em cache por 7 dias
- ✅ **Tokens**: Os tokens de acesso são válidos e armazenados em cache por 24 horas
- ✅ **Fallback**: Se a API falhar, usa placeholders automaticamente
- ⚠️ **Limites**: A API IGDB tem limite de 4 requisições por segundo
- 🔒 **Segurança**: Nunca commit o arquivo `secrets.toml` no Git (já está no .gitignore)

## Exemplo de .gitignore

Adicione ao seu `.gitignore`:

```
.streamlit/secrets.toml
*.pyc
__pycache__/
```

## Troubleshooting

### Imagens não carregam
- Verifique se as credenciais estão corretas em `secrets.toml`
- Confirme que o arquivo está em `.streamlit/secrets.toml`
- Reinicie o aplicativo Streamlit

### Erro de autenticação
- Gere um novo Client Secret no console da Twitch
- Atualize o arquivo `secrets.toml`
- Limpe o cache: pressione "C" no terminal do Streamlit

### Placeholders aparecem
- Significa que as credenciais não estão configuradas
- Siga os passos acima para obter e configurar as credenciais
