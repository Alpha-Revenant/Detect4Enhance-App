import { View, Text, AppState } from "react-native";
import React, { useEffect, useState } from "react";
import {
  NavigationContainer,
  createNavigationContainerRef,
} from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import ScreenNames from "./routes";
import Login from "../screens/login";
import Register from "../screens/signup";
import Home from "../screens/home";
import DrawerNavigator from "./Drawer";
import { createDrawerNavigator } from "@react-navigation/drawer";
import Detection from "../screens/detection";
import AboutUs from "../screens/aboutus";
import FeatherScreen from "../screens/feature";

const Drawer = createDrawerNavigator();
const Stack = createNativeStackNavigator();
// const navigationRef = createNavigationContainerRef();

// function DrawerStack() {
//   return (
//     <Drawer.Navigator initialRouteName="DrawerStack">
//       <Drawer.Screen name="DrawerStack" component={Home} />
//       {/* Add other drawer screens here */}
//     </Drawer.Navigator>
//   );
// }

export default function Routes() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName={ScreenNames.LOGIN} screenOptions={{ headerShown: false }}>
        <Stack.Screen name={ScreenNames.LOGIN} component={Login} />
        <Stack.Screen name={ScreenNames.SIGNUP} component={Register} />
        <Stack.Screen name={ScreenNames.DETECTION} component={Detection} />
        <Stack.Screen name={ScreenNames.ABOUTUS} component={AboutUs} />
        <Stack.Screen name={ScreenNames.DrawerHome} component={DrawerNavigator} />
        {/* <Stack.Screen name={ScreenNames.FEATURES} component={FeatherScreen} /> */}
      </Stack.Navigator>

    </NavigationContainer>
  );
}
