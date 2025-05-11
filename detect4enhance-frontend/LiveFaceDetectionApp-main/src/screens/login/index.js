import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity } from 'react-native';
import styles from './styles';
import { useNavigation } from '@react-navigation/native';

import { sendPasswordResetEmail, signInWithEmailAndPassword } from 'firebase/auth';

import ScreenNames from '../../route/routes';
import { auth } from '../../firebase';
import LottieView from 'lottie-react-native';
import { ScreenWrapper } from 'react-native-screen-wrapper';
import { AppColors } from '../../utils/AppCollors';
import Button from '../../components/button';
import { Entypo, Fontisto, MaterialIcons } from '@expo/vector-icons';
import FlashMessage, { showMessage } from 'react-native-flash-message';

export default function Login() {
  const navigation = useNavigation()

  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [Error, setError] = useState("");
  const [send, setSend] = useState("");
  const [showEye, setShowEye] = useState(false);

  const handleLogin = () => {
    signInWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        // Signed in
        const user = userCredential.user;
        if (user) {
          showMessage({
            message: "Login Successfully",
            type: "success",
          })
          navigation.navigate(ScreenNames.DrawerHome)
        }
      })
      .catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;

        let customErrorMessage = "";
        if (errorCode == "auth/user-not-found") {
          customErrorMessage = "User not found.\n  Please check your email.";
        } else if (errorCode == "auth/wrong-password") {
          customErrorMessage = "Incorrect password.\n  Please try again.";
        } else if (errorCode == "auth/invalid-email") {
          customErrorMessage = "Invalid-email.\n  Please try again.";
        } else if (errorCode == "auth/invalid-credential") {
          customErrorMessage =
            "Invalid Email or Password.\n  Please try again.";
        } else if (errorCode == "auth/network-request-failed") {
          customErrorMessage = "Network-request-failed.\n Please try again.";
        } else if (errorCode == "auth/weak-password") {
          customErrorMessage =
            "Weak password! \n Password should be at least 6 Characters.";
        } else if (errorCode == "auth/missing-password") {
          customErrorMessage = "Missing password! \n Please write password";
        } else if (errorCode == "auth/missing-email") {
          customErrorMessage = "Missing email! \n Please write email";
        } else {
          customErrorMessage = errorMessage;
        }
        setError(customErrorMessage)
        setTimeout(() => {
          setError("");
        }, 1000);
      });
  };

  const handleForgetPassword = () => {
    if (email) {
      sendPasswordResetEmail(auth, email)
        .then(() => {
          setSend("Email has been send Successfully");
          setTimeout(() => {
            setSend("");
          }, 1000);
        })
        .catch((error) => {
          const errorCode = error.code;
          const errorMessage = error.message;
          setError(errorMessage);
          setTimeout(() => {
            setError("");
          }, 1000);
        });
    } else {
      setError("Please enter valid email");
      setTimeout(() => {
        setError("");
      }, 1000);
    }
  };
  return (
    <ScreenWrapper statusBarColor={AppColors.primary} statusBarStyle="light" scrollType='keyboard'>
      <View style={styles.container}>

        <LottieView style={{ width: 210, height: 210, }} source={require('../../assets/Animation - 1723291367748.json')} autoPlay loop />
        <View style={styles.inputContainer}>
          <View style={{ flexDirection: 'row' }}>
            <Fontisto name="email" size={20} color="black" />
            <Text style={styles.label}>Email</Text>
          </View>
          <View style={styles.passwordInputContainer}>
            <TextInput
              style={styles.input}
              placeholder="Enter your email"
              autoCapitalize="none"
              onChangeText={(e) => setEmail(e)}
            />
          </View>
        </View>

        <View style={styles.inputContainer}>
          <View style={{ flexDirection: 'row' }}>
            <MaterialIcons name="password" size={20} color="black" />
            <Text style={styles.label}>Password</Text>
          </View>
          <View style={styles.passwordInputContainer}>
            <TextInput
              style={styles.input}
              placeholder="Enter your password"
              secureTextEntry={showEye ? false : true}
              onChangeText={(p) => setPassword(p)}
            />
            <TouchableOpacity onPress={() => setShowEye(!showEye)} style={styles.eyeIcon}>
              <Entypo name={showEye ? "eye" : "eye-with-line"} size={24} color="black" />
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity style={styles.forgotPassword} onPress={handleForgetPassword}>
          <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
        </TouchableOpacity>


        <Button label="Login" onPress={handleLogin} />

        <TouchableOpacity style={styles.signupButton} onPress={() => navigation.navigate(ScreenNames.SIGNUP)}>
          <Text style={styles.signupButtonText}>Register</Text>
        </TouchableOpacity>

        <View style={styles.error}>
          <Text
            style={{
              alignItems: "flex-end",
              color: "red",
              textAlign: "right",
              fontSize: 13,
            }}
          >
            {Error}
          </Text>
          {send != "" && <Text style={{ alignItems: "flex-end", textAlign: "right", fontSize: 13, color: "green" }}>{send}</Text>}

        </View>
      </View>
    </ScreenWrapper>
  );
}
