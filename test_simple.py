import streamlit as st

st.title("Simple test")
st.write("If you see this, Streamlit is working!")

if st.button("Test Button"):
    st.success("The button works!") 