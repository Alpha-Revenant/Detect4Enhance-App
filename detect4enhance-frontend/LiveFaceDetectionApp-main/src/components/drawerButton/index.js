import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { height, width } from '../../methord/Dimentions';
import { AppColors } from '../../utils/AppCollors';

export default function DrawerButton({ icon, text, onPress }) {
    return (
        <TouchableOpacity style={styles.container} onPress={onPress}>
            <View style={styles.icon}>
                {icon}
            </View>
            <Text style={styles.text}>{text}</Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    container: {
        marginVertical: height(2),
        flexDirection: 'row',
        backgroundColor: AppColors.primary,
        borderRadius: 20,
        width: width(55),
        alignSelf: 'center',
        height: height(6),
        alignItems: 'center',
        paddingHorizontal: width(4),
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: {
                    width: 0,
                    height: 2
                },
                shadowOpacity: 0.25,
                shadowRadius: 3.84
            },
            android: {
                elevation: 5
            }
        })
    },
    text: {
        fontSize: 16,
        color: '#fff',
        marginLeft: 10
    }
})