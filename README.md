# Personalized-AI-Gmail

## Overview
Personalized-AI-Gmail is an intelligent email assistant designed to enhance productivity by leveraging advanced AI capabilities. The tool integrates seamlessly with Gmail to provide users with a personalized experience, including contextual search, dynamic summaries, and cross-platform integration. Whether you're managing a crowded inbox or searching for key insights buried in your digital workspace, Personalized-AI-Gmail is here to help.

## Features
- **Contextual Search**: Search emails using natural language queries. Examples:
  - "What was the email about the project timeline?"
  - "Find the email from last week with the subject 'Team Updates'."
- **Dynamic Summaries**: Summarize lengthy emails and threads into concise, actionable insights to save time.
- **Personalized Suggestions**: Offers intelligent recommendations based on email content and user behavior.
- **Seamless Gmail Integration**: Integrates with Gmail using secure APIs while respecting user privacy.

## Future Scope
### 1. Contextual Search Across Platforms
Expand the current capabilities to enable semantic search across the user's digital ecosystem, including:
- Gmail
- Google Drive
- Notion
- Slack
- Meeting notes, bookmarks, and other productivity tools.

Example Query: "What was that email I received last March about the budget increase?"

### 2. Dynamic Summaries
Enhance summarization to include:
- Summaries of lengthy documents and reports.
- Meeting transcripts distilled into key points and action items.
- Email threads summarized for quick comprehension.

### 3. Cross-Platform Integration
- Connect with popular productivity tools such as:
  - Google Drive
  - Notion
  - Slack
  - Microsoft Teams
- Aggregate and analyze data from multiple platforms in one place.

## Output result comaparision generated by cypher query {AI Generated} vs Vector Store search 

- The traditional generation [ long shot, mostly dependent on the cypher query and structured input]
  
![Traditional Generation](https://github.com/jrspatel/MailMind/blob/main/images/Screenshot%202025-01-06%20231602.png)

- Generated via vector search. Used "WEAVIATE" vector store since it can handle multi modal data i.e. images, emojies, videos and text.
  
![Vector Search](https://github.com/jrspatel/MailMind/blob/main/images/Screenshot%202025-01-06%20232420-%20weaviate.png)

## Improvements actively pursuing 
- Measuring the accuracy of cypher generation by the model. 
- Prompt Evaluation.
- Visualizing the email relations [neo4j map], probably use image models to     visualize the aspects like count of mails per day, with specific person , marketed mails
- Date issues with both the generations.

- caching prompt responses for future use.
- Exploring the integration of scraping techniques to retrieve email data after API fetching, with a focus on preventing any data loss.


