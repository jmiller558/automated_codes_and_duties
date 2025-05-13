from langchain_core.prompts import ChatPromptTemplate

class DeepSelector:
    def __init__(self, llm, logger, code_extractor, agent_actions):
        self.logger = logger

        self.system_prompt = """You are a helpful assistant that can answer questions about the Harmonized Tariff Schedule (HTS) of the United States. The HTS system is used by U.S. Customs and Border Protection (CBP) to determine the duties and taxes that apply to imported goods. You will be provided with a product description, and you will help identify its relevant HTS code.

Your colleague has already determined the most likely four digit codes based on the product description: 
{product_description}

They came to the following conclusion: 

{four_digit_code_response}

Now we need to go a step deeper, and select the full code. 

Here are the relevant codes:

{relevant_data}

Consider:
- Is it reasonable that someone would have described the product this way if it belongs under this code?
- What are the potential uses of the product? Could it belong under another code based on how it is used? This is an important way in which a product could be classified differently than it is described
- What type of goods are most commonly imported? 
- Finally, consider the duty rates. Additional options are only helpful if they have a lower duty rate. DO NOT select options that are less likely and also have a higher duty rate.


Overall, assume the product is as described. But don't rule out a code if you can make a reasonable argument for why it belongs.
"""

        self.human_prompt = """Please select the 6 most likely HTS codes for the follow product: {product_description}
Remember you MUST choose 6 codes, even if you are not sure.
"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", self.human_prompt),
            ]
        )

        self.agent_deploy = self.prompt | llm.with_retry(stop_after_attempt=3)

        self.code_extractor = code_extractor

        self.agent_actions = agent_actions

    async def select_full_codes(self, state):
        """
        Selects the most relevant full HTS codes for a given product description.

        Args:
            state (dict): Contains the product description, response, and relevant data.

        Returns:
            str: The selected HTS codes.
        """
        tag = [state["product_description"]]
        try:
            full_code_options = self.agent_actions.get_full_code_options(state['four_digit_code_list'], tags=tag)
            
            full_code_response = await self.agent_deploy.ainvoke({
                "product_description": state["product_description"],
                "four_digit_code_response": state["responses"][-1],
                "relevant_data": full_code_options
            })
            self.logger.info(f"Selected full codes: {full_code_response}", _tags=tag)
            
            full_code_list = await self.code_extractor.extract_full_codes(full_code_response, tags=tag)

            return {"responses": full_code_response, "full_code_list": full_code_list}
        except Exception as e:
            self.logger.error(f"Error selecting deep HTS codes: {e}")
            raise e