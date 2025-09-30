import streamlit as st


st.title("this is the title of my page")

importbutton = st.button("import")

st.text_input("Enter you text here")

st.text_area("enter ur text")

if importbutton:
    st.success("you just clicked the import button")