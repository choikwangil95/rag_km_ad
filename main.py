import streamlit as st

from src.functions.init import init
from src.functions.ui import (
    print_messages,
    print_sidebar,
    print_user_input,
    print_user_input_with_agent,
)

chain = init()

# 타이틀
st.title("카카오모빌리티 광고상품 챗봇 💬")

# 사이드바 생성
print_sidebar()

# 이전 대화 출력
print_messages()

# 유저 인풋
# print_user_input()
print_user_input_with_agent()
