import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, FlatList } from 'react-native';
import { ScreenWrapper } from 'react-native-screen-wrapper';
import { useNavigation } from '@react-navigation/native';
import { AppColors } from '../../utils/AppCollors';
import CustomHeader from '../../components/header';
import { height, width } from '../../methord/Dimentions';
import { Feather } from '@expo/vector-icons';
export default function FeatureScreen() {
    const navigation = useNavigation();

    // Define your features here
    const features = [
        { id: 1, name: 'Quiz', icon: 'file-text', screen: 'QuizScreen' },
        { id: 2, name: 'Reminder', icon: 'clock', screen: 'ReminderScreen' },
        { id: 3, name: 'Profile', icon: 'user', screen: null },
        { id: 4, name: 'Settings', icon: 'settings', screen: null },
    ];

    // Function to render each feature item
    const renderFeatureItem = ({ item }) => (
        <TouchableOpacity
            style={styles.card}
            onPress={() => item?.screen ? navigation.navigate(item.screen) : null}
        >
            <Feather name={item.icon} size={28} color={AppColors.white} />
            {/* <Icon name={item.icon} size={32} color={AppColors.white} /> */}
            <Text style={styles.cardText}>{item.name}</Text>
        </TouchableOpacity>
    );

    return (
        <ScreenWrapper statusBarColor={AppColors.primary}>
            <View style={styles.container}>
                <CustomHeader navigation={navigation} />

                <View style={styles.headerView}>
                    <Text style={styles.headerText}>App Features</Text>
                </View>

                {/* Feature Cards */}
                <FlatList
                    data={features}
                    renderItem={renderFeatureItem}
                    keyExtractor={(item) => item.id.toString()}
                    contentContainerStyle={styles.featureList}
                    numColumns={2}
                />
            </View>
        </ScreenWrapper>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.primeryLight,
    },
    headerView: {
        alignSelf: 'center',
        marginTop: height(3),
        marginBottom: height(3),
        borderBottomWidth: 1,
        borderColor: "#fff"
    },
    headerText: {
        padding: 15,
        fontSize: 28,
        // fontWeight: 'bold',
        color: AppColors.white,
    },
    featureList: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingBottom: height(5),
    },
    card: {
        flexDirection: 'row',
        backgroundColor: AppColors.primary,
        width: width(40),
        height: height(20),
        margin: width(2),
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        elevation: 5,
    },
    cardText: {
        marginLeft: width(2),
        marginTop: height(1),
        fontSize: 16,
        color: AppColors.white,
        fontWeight: '600',
    },
});

