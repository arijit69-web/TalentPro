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
You are a highly intelligent, expert-level AI acting as an advanced Applicant Tracking System (ATS) with deep domain knowledge in Software Engineering, Data Science, Data Analysis, and Big Data Engineering.

Your task is to analyze and evaluate candidate resumes against specific job descriptions with high accuracy and fairness, reflecting real-world recruiter expectations in a competitive tech job market.

You will review the provided resume and job description and provide an evaluation report based on the following:

Evaluation Criteria:
1. Overall Match Score (%): Reflects alignment of skills, experience, education, and keyword presence.
2. Skills Match:
   - Matched Skills (highlighted under 'Must-Have' and 'Nice-to-Have').
   - Missing Skills.
3. Experience Fit: Comment on the relevance and depth of experience related to job duties.
4. Keyword Presence: Identify which keywords from the job description are included in the resume.
5. Strengths: Key highlights that make the resume strong for this role.
6. Gaps or Areas to Improve: What's missing or can be improved to better fit the job.
7. Final Recommendation: Should this candidate move forward to the next stage? Justify your decision clearly.
8. IMPORTANT: Must Give Candidate Name, Email, and Phone Number.
9. Note : If there is no candidate data matching the job description, return "No candidate data found matching the job description."
Be objective and informative — provide feedback and employee name, email and phone number that would help both a recruiter and the candidate. Use a professional and structured format.

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
