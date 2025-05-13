import json

class FinalSelector:
    def __init__(self, llm, logger, agent_actions):
        self.logger = logger
        self.llm = llm
        self.agent_actions = agent_actions

    async def select_final_codes(self, state):
        """
        Selects final HTS codes for a given product description.

        Args:
            state (dict): Contains the product description, response, and relevant data.

        Returns:
            dict: A dictionary with the selected HTS codes.
        """
        tag = [state["product_description"]]

        full_code_options = self.agent_actions.get_full_code_options(state['full_code_list'], four_digits=False, tags=tag)

        prompt = f"""You are a helpful assistant that can answer questions about the Harmonized Tariff Schedule (HTS) of the United States. The HTS system is used by U.S. Customs and Border Protection (CBP) to determine the duties and taxes that apply to imported goods. You will be provided with a product description, and you will identify its relevant HTS code.

Your colleague has already determined the 6 most likely codes based on the product description: 
{state["product_description"]}.

They came to the following conclusion: 

{state["responses"][-1]}

Here are the details of the codes they selected:
{full_code_options}

Now is the really tough part, you need to select the most likely final code.

Consider:
- What type of goods are most commonly imported? 
- If an attribute is not mentioned in the description, how likely is it that the item possesses this characteristic? If it's not highly likely, then we should probably look for an "other" bucket within the broader category.
- How likely is it that the product might fall into this category? For example if someone describes an item as a "screw", it could possibly also be a bolt or a nut because many people are not sure of the difference. However, if someone describes an item as a drinking glass, it's very unlikely that it is a flower vase, because most people can clearly tell the difference between the two.
- What are the potential uses of the product? This can help in identifying the correct HTS codes.

In addition, I want you to select one more code, this should be the second most likely code, but it needs to have a lower duty rate. So if the duty rate is equal or higher than the first code, that is not helpful, instead move to the next most likely.

Return a dictionary with most_likely_code: YOUR_SELECTION, most_likely_lower_rate_code: YOUR_SECOND_SELECTION 
"""

        output_structure = {
            'type': 'OBJECT',
            'properties': {
                'most_likely_code': {'type':'STRING'},
                'most_likely_lower_rate_code': {'type':'STRING'},
            },
            'required': ['most_likely_code', 'most_likely_lower_rate_code'],
        }

        try:
            response = await self.llm.ainvoke(prompt,
            generation_config={"response_mime_type":'application/json',
            "response_schema": output_structure})
            
            final_codes = json.loads(response.content)

            self.logger.info(f"Selected final codes: {final_codes}", _tags=tag)
            return {"final_codes": final_codes}
        except Exception as e:
            self.logger.error(f"Error selecting final HTS codes: {e}")
            raise e