from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


def generate_restaurant_idea(cuisine, google_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=0.7,
    )

    template_name = "I want to open a restaurant for {cuisine} food, suggest exactly one fancy name for it."
    prompt_name = PromptTemplate(template=template_name, input_variables=["cuisine"])
    chain_name = LLMChain(
        llm=llm, prompt=prompt_name, verbose=True, output_key="restaurant_name"
    )

    template_menu = "Suggest some menu items for {restaurant_name}. Return them as a comma separated list."
    prompt_menu = PromptTemplate(
        template=template_menu, input_variables=["restaurant_name"]
    )
    chain_menu = LLMChain(
        llm=llm, prompt=prompt_menu, verbose=True, output_key="menu_items"
    )

    # a SimpleSequentialChain would not return the restaurant name (intermediate variable); thus using SequentialChain
    chain = SequentialChain(
        chains=[chain_name, chain_menu],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
    )

    output = chain.invoke({"cuisine": cuisine})
    return output["restaurant_name"], output["menu_items"]


def main():
    st.set_page_config(page_title="Restaurant idea generator", page_icon="🍴")
    st.title("Restaurant idea generator")

    google_api_key = st.sidebar.text_input("Paste Google generative AI API key")
    cuisine = st.sidebar.selectbox(
        "Pick a cuisine",
        (
            "American",
            "Caribbean",
            "Chinese",
            "French",
            "Indian",
            "Italian",
            "Japanese",
            "Mexican",
        ),
    )

    if cuisine and google_api_key:
        restaurant_name, menu_items = generate_restaurant_idea(cuisine, google_api_key)
        st.header(restaurant_name.strip())
        menu_items_list = menu_items.strip().split(",")
        st.write("**Menu Items**")
        for item in menu_items_list:
            st.write("-", item)


if __name__ == "__main__":
    main()
