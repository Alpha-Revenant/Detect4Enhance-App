// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from 'firebase/firestore';
import { initializeAuth, getReactNativePersistence } from 'firebase/auth';
import ReactNativeAsyncStorage from '@react-native-async-storage/async-storage';

import { getAuth } from "firebase/auth";
import { getStorage } from "firebase/storage";
import { getDatabase } from "firebase/database";
// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDHZ0bRztSiIWaJzf2KZms3rDLXa5sFooA",
  authDomain: "realtimefacedetectionapp.firebaseapp.com",
  projectId: "realtimefacedetectionapp",
  storageBucket: "realtimefacedetectionapp.appspot.com",
  messagingSenderId: "52926753972",
  appId: "1:52926753972:web:5c1e513a2571aae15b7be1"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(ReactNativeAsyncStorage),
});
const storage = getStorage(app)
const db = getFirestore(app);
const database = getDatabase(app)
export { auth }
export { db }
export { storage }
export { database }