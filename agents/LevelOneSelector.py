from langchain_core.prompts import ChatPromptTemplate

class LevelOneSelector:
    def __init__(self, llm, logger, code_extractor, agent_actions):
        self.logger = logger

        self.system_prompt = """You are a helpful assistant that can answer questions about the Harmonized Tariff Schedule (HTS) of the United States. The HTS is a system for classifying goods imported into the United States. It is used by U.S. Customs and Border Protection (CBP) to determine the duties and taxes that apply to imported goods. You will be provided with a product description, and you will help identify its relevant HTS code.

Your colleague has already determined the most likely chapters to consult based on the product description: 
{product_description}

They came to the following conclusion: {chapter_response}

Here are the relevant codes within these chapters

{relevant_data}

Select up to 6 of the most likely HTS codes. 

Consider:
- Is it reasonable that someone would have described the product this way if it belongs under this code?
- What type of goods are most commonly imported? 
- What are the potential uses of the product? This can help in identifying the correct HTS codes.

Overall, assume the product is as described. But don't rule out a code if you can make a reasonable argument for why it belongs.
"""
        
        self.human_prompt = """Select up to 6 of the most likely HTS codes for the following product: {product_description}."""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", self.human_prompt),
            ]
        )

        self.agent_deploy = self.prompt | llm.with_retry(stop_after_attempt=3)

        self.code_extractor = code_extractor

        self.agent_actions = agent_actions

    async def select_four_digit_codes(self, state):
        """
        Selects the most relevant HTS codes for a given product description.

        Args:
            state (dict): Contains the product description, response, and relevant data.

        Returns:
            str: The selected HTS codes.
        """
        tag = [state["product_description"]]
        try:
            four_digit_code_options = self.agent_actions.get_four_digit_code_options(state['chapters_list'], tags=tag)
            
            initial_code_response = await self.agent_deploy.ainvoke({
                "product_description": state["product_description"],
                "chapter_response": state["responses"][-1],
                "relevant_data": four_digit_code_options,
            })
            self.logger.info(f"Selected initial codes: {initial_code_response}", _tags=tag)
            
            four_digit_code_list = await self.code_extractor.extract_four_digit_codes(initial_code_response, tags=tag)
            
            return {"responses": initial_code_response, "four_digit_code_list": four_digit_code_list}
        except Exception as e:
            self.logger.error(f"Error selecting HTS codes: {e}")
            raise e