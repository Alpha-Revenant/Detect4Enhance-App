import React, { useEffect, useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons'; // Assuming you're using Expo for icons
import styles from './styles';
import { useNavigation } from '@react-navigation/native';
export default function CustomHeader({ navigation }) {
  // const navigation = useNavigation()

  const openDrawer = () => {
    navigation?.openDrawer()
  }
  return (
    <View style={styles.header}>
      <View style={styles.headerContainer}>
        <TouchableOpacity onPress={openDrawer}>
          <Ionicons name="menu-outline" size={30} color="#fff" style={styles.menuIcon} />
        </TouchableOpacity>
        <Text style={styles.detectionText}>Class Room Pulse</Text>
      </View>
    </View>
  );
};
