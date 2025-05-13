from langchain_core.prompts import ChatPromptTemplate

class ChapterSelector:
    def __init__(self, llm, logger, chapters_list, code_extractor):

        self.logger = logger

        self.chapters_list = chapters_list

        self.system_prompt = """You are a helpful assistant that can answer questions about the Harmonized Tariff Schedule (HTS) of the United States. The HTS is a system for classifying goods imported into the United States. It is used by U.S. Customs and Border Protection (CBP) to determine the duties and taxes that apply to imported goods. You will be provided with a product description, and you will need to help identify its relevant HTS code. The first step will be to determine the correct chapters to search for the HTS code.
        
Here is a list of chapters and their descriptions:
        
{chapters_list}
        
Select the 3 most likely chapters, and be sure to provide your reasoning. Consider what are the most common or likely chapters and make sure to select at least 3 options, since this is the first step we don't want our search to be too narrow at the beginning."""

        self.human_prompt = """Based on the list of chapters provided, please select the 3 most likely chapters to consult for the HTS code for the follow product: {product_description}."""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", self.human_prompt),
            ]
        )

        self.agent_deploy = self.prompt | llm.with_retry(stop_after_attempt=3)

        self.code_extractor = code_extractor

    async def select_chapters(self, state):
        """
        Selects the most relevant chapters for a given product description.

        Args:
            state (dict): Contains the product description and a list of chapters.

        Returns:
            str: The selected chapters.
        """
        tag = [state["product_description"]]
        try:
            chapter_response = await self.agent_deploy.ainvoke({
                "product_description": state["product_description"],
                "chapters_list": self.chapters_list
            })
            self.logger.info(f"Selected chapters: {chapter_response}", _tags=tag)

            chapter_list = await self.code_extractor.extract_chapters(chapter_response, tags=tag)
            
            return {"responses": chapter_response, "chapters_list": chapter_list}
        except Exception as e:
            self.logger.error(f"Error selecting chapters: {e}")
            raise e