import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { ScreenWrapper } from 'react-native-screen-wrapper';
import { useNavigation } from '@react-navigation/native';
import { AppColors } from '../../utils/AppCollors';
import CustomHeader from '../../components/header';
import { height, width } from '../../methord/Dimentions';
import { AntDesign } from '@expo/vector-icons';

export default function QuizScreen() {
    const navigation = useNavigation();

    // Define your quiz topics
    const quizzes = [
        { id: 1, name: 'Programming fundamental', icon: 'book', color: AppColors.primary, screen: 'TakeQuiz' },
        { id: 2, name: 'Database', icon: 'rocket1', color: AppColors.primary, screen: 'TakeQuiz' },
        { id: 3, name: 'Data Structures', icon: 'calculator', color: AppColors.primary, screen: 'TakeQuiz' },
    ];

    // Function to render each quiz button
    const renderQuizButton = (quiz) => (
        <TouchableOpacity
            key={quiz.id}
            style={[styles.quizButton, { backgroundColor: quiz.color }]}
            onPress={() => navigation.navigate(quiz.screen, { QuizOf: quiz.name })}
        >
            <View style={styles.quizItemView}>

                <AntDesign name={quiz.icon} size={28} color="#fff" />
                <Text style={styles.quizButtonText}>{quiz.name}</Text>
            </View>

        </TouchableOpacity>
    );

    return (
        <ScreenWrapper statusBarColor={AppColors.primary}>
            <View style={styles.container}>
                <CustomHeader navigation={navigation} />

                {/* Header Section */}
                <View style={styles.headerView}>
                    <Text style={styles.headerText}>Select Your Quiz</Text>
                </View>

                {/* Quiz Buttons */}
                <View style={styles.quizContainer}>
                    {quizzes.map((quiz) => renderQuizButton(quiz))}
                </View>
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
        marginTop: height(5),
    },
    headerText: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#fff',
    },
    quizContainer: {
        flex: 1,
        // justifyContent: 'center',
        marginTop: height(3),
        alignItems: 'center',
    },
    quizButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        width: width(80),
        height: height(12),
        borderRadius: 25,
        marginVertical: height(2),
        elevation: 8,
    },
    quizItemView: {
        width: width(65),
        justifyContent: 'center',
        alignItems: "center",
        alignSelf: 'center',
        flexDirection: 'row'
    },
    quizButtonText: {
        // paddingVertical: height(2),

        marginLeft: width(3),
        fontSize: 20,
        fontWeight: '600',
        color: '#fff',
        textAlign: "center",

    },
});
