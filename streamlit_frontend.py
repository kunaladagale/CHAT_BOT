# import streamlit as st
# from langgraph_backend import chatbot
# from langchain_core.messages import HumanMessage
#
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}
#
# # ---- session state ----
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []
# if 'generating' not in st.session_state:
#     st.session_state['generating'] = False
#
# st.sidebar.title('Langraph Chat Bot')
#
# st.sidebar.button('New Chat')
#
# st.sidebar.header('My Conversation')
#
# # ---- token-by-token generator for st.write_stream ----
# def stream_reply(user_text):
#     """Yield the assistant's answer token by token as the model produces it."""
#     for chunk, metadata in chatbot.stream(
#         {'messages': [HumanMessage(content=user_text)]},
#         config=CONFIG,
#         stream_mode="messages",   # <-- streams LLM tokens, not whole-graph steps
#     ):
#         if chunk.content:
#             yield chunk.content
#
#
# # ---- render the conversation so far ----
# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.markdown(message['content'])
#
# # ---- input is DISABLED while the model is generating ----
# user_input = st.chat_input('Type here', disabled=st.session_state['generating'])
#
# # Step 1: new message -> store it, lock input, rerun to redraw disabled state
# if user_input and not st.session_state['generating']:
#     st.session_state['message_history'].append({'role': 'user', 'content': user_input})
#     st.session_state['generating'] = True
#     st.rerun()
#
# # Step 2: generating -> stream the answer live, then unlock
# if st.session_state['generating']:
#     last_user = st.session_state['message_history'][-1]['content']
#     with st.chat_message('assistant'):
#         try:
#             # st.write_stream types tokens out live AND returns the full text
#             ai_message = st.write_stream(stream_reply(last_user))
#         except Exception as e:
#             ai_message = f"⚠️ Error: {e}"
#             st.markdown(ai_message)
#
#     st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
#     st.session_state['generating'] = False
#     st.rerun()