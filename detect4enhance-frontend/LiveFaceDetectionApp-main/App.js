import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';

import { GestureHandlerRootView } from "react-native-gesture-handler";

import { NavigationContainer } from '@react-navigation/native';
import Login from './src/screens/login';
import Routes from './src/route';
import FlashMessage from 'react-native-flash-message';
import ExpoNotification, { NotificationContext } from './src/notificationContext';

export default function App() {
  return (
    <ExpoNotification>
      <GestureHandlerRootView style={{ flex: 1 }}>
        {/* <DrawerNavigator /> */}
        {/* <Login /> */}


        <Routes />
        <FlashMessage position="top" />
      </GestureHandlerRootView>
    </ExpoNotification>

  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
