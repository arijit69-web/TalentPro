const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const pdfParse = require('pdf-parse');
const { DataAPIClient } = require('@datastax/astra-db-ts');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const OpenAI = require('openai');
const axios = require('axios');

dotenv.config();

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  OPENAI_API_KEY,
  PORT = 3001,
  GITHUB_TOKEN
} = process.env;

const app = express();
app.use(express.json());
app.use(cors());
app.use(fileUpload());

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const createCollection = async () => {
  try {
    const collections = await db.listCollections();
    const exists = collections.some(c => c.name === ASTRA_DB_COLLECTION);
    if (!exists) {
      await db.createCollection(ASTRA_DB_COLLECTION, {
        vector: {
          dimension: 1536,
          metric: 'dot_product',
        },
      });
      console.log('Collection created');
    }
  } catch (err) {
    console.error('Error creating collection:', err);
  }
};

app.post('/upload-resume', async (req, res) => {
  try {
    const { name, role, githubUsername } = req.body;
    const file = req.files?.resume;

    if (!file || !name || !role || !githubUsername) {
      return res.status(400).json({ error: 'Resume file, name, role, and GitHub username are required.' });
    }

    const reposResponse = await axios.get(
      `https://api.github.com/users/${githubUsername}/repos?per_page=100`,
      {
        headers: {
          Authorization: `Bearer ${GITHUB_TOKEN}`,
          Accept: 'application/vnd.github+json',
        },
      }
    );
    const repos = reposResponse.data;

    const languagesSet = new Set();

    for (const repo of repos) {
      if (repo.language) {
        languagesSet.add(repo.language);
      }
    }

    const skills = Array.from(languagesSet);

    const pdfData = await pdfParse(file.data);
    const resumeText = pdfData.text;

    const chunks = await splitter.splitText(resumeText);
    const collection = await db.collection(ASTRA_DB_COLLECTION);

    for (const chunk of chunks) {
      const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: chunk,
        encoding_format: 'float',
      });

      const vector = embedding.data[0].embedding;

      await collection.insertOne({
        $vector: vector,
        name,
        role,
        skills,
        text: chunk,
      });
    }

    res.json({ status: 'success', message: 'Resume uploaded and stored successfully.', extractedSkills: skills });
  } catch (error) {
    console.error(error?.response?.data || error.message || error);
    res.status(500).json({ status: 'error', message: 'Failed to upload resume.' });
  }
});

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'Server is Running', timestamp: new Date().toISOString() });
});

app.post('/query', async (req, res) => {
  try {
    const { messages } = req.body;
    const latestMessage = messages[messages.length - 1]?.content;
    let docContext = "";

    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: latestMessage,
      encoding_format: "float",
    });

    try {
      const collection = await db.collection(ASTRA_DB_COLLECTION);
      const cursor = collection.find(null, {
        sort: { $vector: embedding.data[0].embedding },
        limit: 10,
      });

      const documents = await cursor.toArray();
      const docsMap = documents.map(doc => doc.text);
      docContext = JSON.stringify(docsMap);
    } catch (err) {
      console.error("Error fetching documents:", err);
      docContext = "";
    }

    const prompt = `
You are a highly advanced AI Applicant Tracking System (ATS) with expert-level knowledge in Software Engineering, Data Science, Data Analysis, and Big Data Engineering.

Your task is to evaluate a candidate's resume against a specific job description and generate a detailed, recruiter-style report. Be objective, fair, and precise — reflecting real-world technical hiring expectations.

Evaluation Criteria:
1. Overall Match Score (%) - Based on alignment of skills, experience, and education.
2. Skills Match:
   - Matched Skills 
   - Missing Skills
3. Final Recommendation - Should the candidate move to the next stage? Clearly justify.
4. Candidate Information - Full Name, Email, Phone Number (must be included).
5. Note - If no relevant candidate data matches the job, respond with:
   "No candidate data found matching the job description."
If the candidate's full name, email, and phone number cannot be retrieved, do not include any candidate details in the report. Instead, respond with:
"No candidate data found matching the job description."
----------
START CONTEXT
${docContext}
END CONTEXT
----------
QUESTION: ${latestMessage}
----------
`;


    const chatResponse = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'system', content: prompt },
        ...messages,
      ],
      temperature: 0.7,
    });

    const reply = chatResponse.choices[0].message.content;
    const cleanReply = reply.replace(/\n/g, ' ');
    res.json({ response: cleanReply });
  } catch (error) {
    console.error("Error querying ATS bot:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

createCollection().then(() => {
  app.listen(PORT, () => {
    console.log(`ATS API running at http://localhost:${PORT}`);
  });
});
