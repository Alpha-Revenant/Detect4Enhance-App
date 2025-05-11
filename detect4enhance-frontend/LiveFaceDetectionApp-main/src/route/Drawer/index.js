import { useState, useEffect } from 'react';
import { Text, View, TouchableOpacity, Button, useWindowDimensions, StyleSheet, Image, Alert } from 'react-native';

import { DrawerActions, NavigationContainer, useFocusEffect, useIsFocused, useNavigation } from '@react-navigation/native';

import { createDrawerNavigator, useDrawerStatus } from '@react-navigation/drawer';
import { createStackNavigator } from '@react-navigation/stack';
import { DrawerContentScrollView, DrawerItem } from '@react-navigation/drawer';
import { AntDesign, Ionicons, Feather } from '@expo/vector-icons';
import ScreenNames from '../routes';
import styles from './styles';
import { signOut } from 'firebase/auth';
import { auth, db } from '../../firebase';
import { doc, getDoc } from 'firebase/firestore';
import Home from '../../screens/home';
import DrawerButton from '../../components/drawerButton';
import FeatherScreen from '../../screens/feature';
import QuizScreen from '../../screens/quiz';
import TakeQuiz from '../../screens/takeQuiz';
import ReminderScreen from '../../screens/remainder';
const Drawer = createDrawerNavigator();



function CustomDrawerContent(props) {
  const [userName, setUserName] = useState("");

  const isfocus = useIsFocused()
  const status = useDrawerStatus()
  // useFocusEffect(() => {
  //   getDataFromFirestore()
  // })
  useEffect(() => {
    if (auth?.currentUser?.uid) getDataFromFirestore()

  }, [status])

  const getDataFromFirestore = async () => {
    const docRef = doc(db, "users", auth?.currentUser?.uid);
    try {
      const docSnap = await getDoc(docRef);

      if (docSnap.exists()) {
        setUserName(docSnap.data()?.username)
      } else {
        console.log("No such document!");
      }
    } catch (error) {
      console.log("error", error)
    }
  }

  const navigation = useNavigation()

  const handleLogout = () => {
    // Show confirmation alert before logging out
    Alert.alert(
      'Confirm Logout',
      'Are you sure you want to log out?',
      [
        {
          text: 'Cancel',
          style: 'cancel'
        },
        {
          text: 'Logout',
          onPress: () => { userLogout() }
        }
      ],
      { cancelable: false }
    );
  };

  const userLogout = () => {
    signOut(auth).then(() => {

      // alert('Sign-out successful')
      navigation.navigate(ScreenNames.LOGIN)
    }).catch((error) => {
      // An error happened.
      alert('An error happened')
    });

  }
  const navigateAndCloseDrawer = (screen) => {
    navigation.navigate(screen);
    navigation.dispatch(DrawerActions.closeDrawer());
  };

  return (
    <View style={styles.drawerContainer}>
      <View style={styles.header}>
        <Image source={require('./../../assets/logo.png')} resizeMode="contain" style={styles.logo} />
        <Text style={styles.nameText} numberOfLines={1}>{userName ? userName : ""}</Text>
      </View>

      <DrawerButton icon={<AntDesign name="home" size={24} color="#fff" />} text="Home" onPress={() => navigateAndCloseDrawer(ScreenNames.HOME)} />
      <DrawerButton icon={<Feather name="info" size={24} color="#fff" />} text="About Us" onPress={() => navigateAndCloseDrawer(ScreenNames.ABOUTUS)} />
      <DrawerButton icon={<Feather name="star" size={24} color="#fff" />} text="Features" onPress={() => navigateAndCloseDrawer(ScreenNames.FEATURES)} />

      <View style={styles.content}>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <AntDesign name="logout" size={24} color="#fff" />
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}


export default function DrawerNavigator() {

  return (
    <Drawer.Navigator drawerContent={(props) => <CustomDrawerContent {...props} />}

      screenOptions={{
        headerShown: false,
        headerTitleAlign: 'right',
        headerStyle: {
          backgroundColor: '#2c2c6c', // Instagram's header color
        },
        headerTintColor: '#fff', // Instagram's header text color
        headerTitleStyle: {
          fontWeight: 'bold',
        },

      }}>

      <Drawer.Screen name={ScreenNames.HOME} component={Home} />
      <Drawer.Screen name={ScreenNames.FEATURES} component={FeatherScreen} />
      <Drawer.Screen name='QuizScreen' component={QuizScreen} />
      <Drawer.Screen name='TakeQuiz' component={TakeQuiz} />
      <Drawer.Screen name='ReminderScreen' component={ReminderScreen} />
    </Drawer.Navigator>
  )
}