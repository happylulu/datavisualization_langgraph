// pages/api/uploadCsv.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { storage, firestore } from '../../config/firebaseConfig';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { collection, addDoc } from 'firebase/firestore';

type Data = {
    message?: string;
    url?: string;
    docId?: string;
    error?: string;
};

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse<Data>
) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Only POST requests are allowed' });
    }

    const { file, filename } = req.body;

    if (!file || !filename) {
        return res.status(400).json({ error: 'File data or filename missing' });
    }

    try {
        // Convert base64 file data to a Uint8Array for Firebase upload
        const fileData = Buffer.from(file.split(',')[1], 'base64');
        const storageRef = ref(storage, `uploads/${filename}`);

        // Upload the file to Firebase Storage
        const snapshot = await uploadBytes(storageRef, fileData, {
            contentType: 'text/csv',
        });

        // Get the download URL for the uploaded file
        const downloadURL = await getDownloadURL(snapshot.ref);

        // Save file metadata to Firestore
        const docRef = await addDoc(collection(firestore, 'agent_csv_files'), {
            name: filename,
            url: downloadURL,
            createdAt: new Date(),
        });

        res.status(200).json({
            message: 'File uploaded successfully',
            url: downloadURL,
            docId: docRef.id,
        });
    } catch (error: any) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
}
