# gs24-ai-workshop
Repository for development of InterSystems Global Summit 2024 Gen AI workshop

## Setup/Introduction

Objectives: In this section, participants will understand the structure of the workshop, what they will achieve by the end and learn about tools and libraries they'll be using.

Actions to take:

1. **Introduce the workshop**: Briefly explain the concepts of Generative AI and Vector Search. Show how these can be applied in real-world scenarios like Finance and Healthcare.
2. **Setup guide**: Use Instruqt or provide a link where the user can get started with Iris and gain familiarity with it. Introduce key Python libraries as well.

## Loading Data into InterSystems IRIS

Objectives: In this section, participants will learn how to use Iris for efficiently loading data and perform vector searches.

Actions to take:

1. **Analyze and load data**: Depending on the use case of choice (Finance or Healthcare), participants will load a dataset into Iris. There might be two different paths here: one for people who are already familiar with Iris and can start loading data directly from their computers and another for those who need to familiarize themselves with Iris first by using pre-loaded data in the Iris container provided in Instruqt.
2. **Use InterSystems IRIS**: Any necessary setup like creating vector tables and how to insert data into vector columns.

## Vector Search for Data Retrieval

Objectives: In this section, participants will learn how to use Iris for efficient information retrieval from datasets. It includes executing both basic and hybrid (filtered) queries.

Actions to take:

1. **Learn Vector Search**: Define what Vector Search is, and how it operates on multi-dimensional data.
2. **Operations in InterSystems IRIS**: Show how to execute basic and hybrid queries using Iris. Give examples of combining traditional SQL filters with vector similarity measures for targeted data retrieval.

## Debugging / Common Errors

Objectives: Provide common debugging techniques and possible error messages that participants might encounter while following the Steps 2 or 3.

Actions to take:

1. **Debugging Techniques**: Discuss troubleshooting methods for some common issues that participants may come across when setting up their environment and using Iris for data retrieval.

## Connecting to a Large Language Model (LLM) for Retrieval Augmented Generation (RAG)

Objectives: In this section, participants learn about the RAG architecture for combining Vector Search with LLM, and then finally use it in practice.

Actions to take:

1. **Learn the RAG Architecture**: Understand the importance of the RAG architecture, which combines the power of sorting according to relevance with the natural language generation capabilities of an LLM.
2. **Use LlamaIndex**: With theformed Learnings on our Feelings, their role in Simplifying the Retrieval, and Augmentation Process for RAG Applications is encouraged.

## Putting It All Together

Objectives: The aim of this step is to enable users to build an app with these tools that would return appropriate data based on a Natural Language Query, as per the RAG architecture principles introduced up to this point.

Actions to take:

1. **Test the App**: Build an app or some sort of Unit Test that would check if the model's data retrieval and generation functions are appropriate and flowing smoothly.

## Bonus Exercises

### From Existing Data, How to Vectorize?

In this section we will discuss methods to convert an existing dataset into a form we can search using vectors.

## Using Local LLMs Instead of OpenAI via API Key

This exercise will provide a method to use local language models instead of the OpenAI model.

```py
print("Coming Soon!")
```