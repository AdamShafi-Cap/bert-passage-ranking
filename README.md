# Bert Paragraph Search and Summarization

## Build With Docker

docker build . -t bert-passage

docker run -d -p 8501:8501 bert-passage

## Using the app

There are 3 ways to send information to the app:

### Use the Draft Brexit Trade Agreement

This is the fastest way to explore the functionality.
Because this document has separate ‘articles’, we are able to split this into larger chunks which work well for the model. We have also preprocessed this document as it is large (over 200 pages). This means the response is fast from within the app and you can switch between models very easily. 

### Upload a PDF

The app will convert the PDF to text and split this into paragraphs. In order to do this, it uses new lines in the documents. The text will only split where there are duplicate newlines (ie a big space) to try to simulate new paragraphs. Depending on the formatting of the document, this can lead to lots of short paragraphs.
You can opt to remove the first few pages of your document to avoid preamble or contents pages.
Once the document has been uploaded, the model needs to process it. In a production system, this would happen just once, before any users need it. In this lightweight app, this has to happen once you load the document. Recommended document length is around 10-20 pages.
When you select a model, it will process the document. You can now enter your queries to return relevant results.

### Upload a csv file

This skips the need to process the PDF and allows more control of paragraph sizes. To do this, just upload a csv with 1 row per paragraph and no column titles/headers. Your paragraphs should be in the first column of the file.
As for PDFs, your file will be processed once you select a model.

### Models used

This app demonstrates BERT’s application to 2 tasks in the field of Natural Language Processing.

Task 1: Given a query, can BERT surface relevant sections of a document (searching)?
Task 2: Given some text, can BERT generate an accurate summary using sentences from the text?

To do this, multiple BERT models are used. As BERT is open-sourced, researchers have created BERT variations (eg ‘RoBERTa’) and fine-tuned versions of many models are available.

This app uses 3 BERT variations.

1.	Regular/Base BERT: BERT is an open-source model to process natural language developed by Google. It was designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.
2.	DistillBERT: Developed by Huggingface as a faster, more efficient version of BERT using a compression technique called distillation
3.	RoBERTa: Developed by Facebook as a larger, more optimized version of BERT. When released, this model improved BERT’s scores on a variety of benchmarks.

Regular BERT was trained on over 3m words from a dataset of books and English Wikipedia. In this app, fine tuned versions of the above models are used.

1.	For RoBERTa and DistillBERT, we have used versions specifically trained on task 1 by researchers at the Technical University of Darmstadt using several Stanford datasets.
2.	We have also used a version of DistillBERT trained by the same researchers on a Microsoft dataset of question and answer pairs. This model should be particularly good at answering questions.

The table below summarises tasks, models and fine tuning.

|  Task	          |  BERT Variation       |  Fine Tuning                  |
|  -------------  |  ------------------   |  ---------------------------  |
|  Summarisation  |	 Base BERT        |  None                         |
|  Searching      |	 DistillBERT      |  Stanford similarity dataset  |
|  Searching      |	 DstillBERT Q&A   |  Stanford similarity dataset  |
|  Searching      |	 RoBERTa (large)  |  Microsoft Q&A Dataset        |







