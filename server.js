const express = require('express');
const dotenv = require('dotenv');
const fileUpload = require('express-fileupload');
const pdfParse = require('pdf-parse');
const cors = require('cors');
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
  PORT = 3000
} = process.env;


const app = express();
app.use(express.json());
app.use(fileUpload());
app.use(cors());

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

    const reposResponse = await axios.get(`https://api.github.com/users/${githubUsername}/repos?per_page=100`);
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

createCollection().then(() => {
  app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
  });
});
