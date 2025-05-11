import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import CustomHeader from '../../components/header';
import { ScreenWrapper } from 'react-native-screen-wrapper';
import { AppColors } from '../../utils/AppCollors';
import LottieView from 'lottie-react-native';
import { height, width } from '../../methord/Dimentions';
import Button from '../../components/button';
import { useNavigation } from '@react-navigation/native';
import ScreenNames from '../../route/routes';

export default function Home() {
    const navigation = useNavigation()
    return (
        <ScreenWrapper statusBarColor={AppColors.primary} barStyle='light-content'>
            <View style={styles.parentView}>
                <CustomHeader navigation={navigation} />
                <View style={styles.itemContainer}>
                    <LottieView style={{ width: 200, height: 200 }} source={require('./../../assets/Animation - 1723300983385.json')} autoPlay loop />
                    <Text style={styles.title}>Detect4Enhance</Text>
                    <Text style={styles.des}>Classroom Pulse is a groundbreaking mobile app that uses facial expression detection to measure students' emotions and engagement in real-time.</Text>
                </View>
                <View style={{ flex: 1, justifyContent: 'flex-end', alignSelf: "center", marginBottom: height(5) }}>
                    <Button label="Get Started" onPress={() => { navigation.navigate(ScreenNames.DETECTION) }} />
                </View>
            </View>
        </ScreenWrapper>
    );
}
const styles = StyleSheet.create({
    parentView: {
        flex: 1,
    },
    itemContainer: {
        marginTop: height(10),
        justifyContent: "center",
        alignItems: "center",

    },
    title: {
        fontSize: 20,
        fontWeight: "bold",

    },
    des: {
        width: width(85),

        marginTop: height(2),
        textAlign: "center",
        fontSize: 16,
        // fontWeight: "500",
    }
})