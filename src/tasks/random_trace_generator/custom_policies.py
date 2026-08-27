FERRY_POLICY_DESCRIPTION= """
[boolean_features]

[numerical_features]
    <remaining> ::= 
    @numerical_count
        @concept_intersection
            @concept_atomic_state "car"
            @concept_negation
                @concept_role_value_map_equality
                    @role_atomic_state "at"
                    @role_atomic_goal "at" true

    <carried> ::= 
    @numerical_count
        @concept_intersection
            @concept_atomic_state "car"
            @concept_atomic_state "on"

  <ready_unload> ::= 
    @numerical_count
        @concept_intersection
            @concept_atomic_state "car"
            @concept_intersection
                @concept_atomic_state "on"
                @concept_existential_quantification
                    @role_atomic_goal "at" true
                    @concept_atomic_state "at-ferry"

   <car_at_ferry> ::= 
    @numerical_count
        @concept_intersection
            @concept_atomic_state "car"
            @concept_intersection
                @concept_negation
                    @concept_role_value_map_equality
                        @role_atomic_state "at"
                        @role_atomic_goal "at" true
                @concept_existential_quantification
                    @role_atomic_state "at"
                    @concept_atomic_state "at-ferry"

[policy_rules]
    { @greater_numerical_condition <ready_unload> }
    -> { @decrease_numerical_effect <remaining>,
         @decrease_numerical_effect <carried>,
         @decrease_numerical_effect <ready_unload> }

    { @greater_numerical_condition <carried>,
      @equal_numerical_condition <ready_unload> }
    -> { @increase_numerical_effect <ready_unload>,
         @unchanged_numerical_effect <carried>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <carried>,
      @greater_numerical_condition <car_at_ferry> }
    -> { @increase_numerical_effect <carried>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <car_at_ferry>,
      @greater_numerical_condition <remaining> }
    -> { @increase_numerical_effect <car_at_ferry>,
         @unchanged_numerical_effect <remaining> }"""


MICONIC_POLICY_DESCRIPTION = """[boolean_features]

[numerical_features]
    <remaining> ::= 
        @numerical_count
            @concept_intersection
                @concept_atomic_state "passenger"
                @concept_negation
                    @concept_atomic_state "served"

    <boarded> ::= 
        @numerical_count
            @concept_atomic_state "boarded"

    <ready_depart> ::= 
        @numerical_count
            @concept_intersection
                @concept_atomic_state "boarded"
                @concept_existential_quantification
                    @role_atomic_state "destin"
                    @concept_atomic_state "lift-at"

    <ready_board> ::= 
        @numerical_count
            @concept_existential_quantification
                @role_atomic_state "origin"
                @concept_atomic_state "lift-at"

    <boarded_dest_above> ::= 
        @numerical_count
            @concept_intersection
                @concept_atomic_state "boarded"
                @concept_existential_quantification
                    @role_atomic_state "destin"
                    @concept_existential_quantification
                        @role_inverse
                            @role_atomic_state "above"
                        @concept_atomic_state "lift-at"

    <boarded_dest_below> ::= 
        @numerical_count
            @concept_intersection
                @concept_atomic_state "boarded"
                @concept_existential_quantification
                    @role_atomic_state "destin"
                    @concept_existential_quantification
                        @role_atomic_state "above"
                        @concept_atomic_state "lift-at"

    <origin_above> ::= 
        @numerical_count
            @concept_existential_quantification
                @role_atomic_state "origin"
                @concept_existential_quantification
                    @role_inverse
                        @role_atomic_state "above"
                    @concept_atomic_state "lift-at"

    <origin_below> ::= 
        @numerical_count
            @concept_existential_quantification
                @role_atomic_state "origin"
                @concept_existential_quantification
                    @role_atomic_state "above"
                    @concept_atomic_state "lift-at"

    <dist_boarded_dest_above> ::= 
        @numerical_distance
            @concept_atomic_state "lift-at"
            @role_atomic_state "above"
            @concept_existential_quantification
                @role_inverse
                    @role_atomic_state "destin"
                @concept_atomic_state "boarded"

    <dist_boarded_dest_below> ::= 
        @numerical_distance
            @concept_atomic_state "lift-at"
            @role_inverse
                @role_atomic_state "above"
            @concept_existential_quantification
                @role_inverse
                    @role_atomic_state "destin"
                @concept_atomic_state "boarded"

    <dist_origin_above> ::= 
        @numerical_distance
            @concept_atomic_state "lift-at"
            @role_atomic_state "above"
            @concept_existential_quantification
                @role_inverse
                    @role_atomic_state "origin"
                @concept_atomic_state "passenger"

    <dist_origin_below> ::= 
        @numerical_distance
            @concept_atomic_state "lift-at"
            @role_inverse
                @role_atomic_state "above"
            @concept_existential_quantification
                @role_inverse
                    @role_atomic_state "origin"
                @concept_atomic_state "passenger"


[policy_rules]
    { @greater_numerical_condition <ready_depart> }
    -> { @decrease_numerical_effect <remaining>,
         @decrease_numerical_effect <boarded>,
         @decrease_numerical_effect <ready_depart> }

    { @greater_numerical_condition <ready_board> }
    -> { @increase_numerical_effect <boarded>,
         @decrease_numerical_effect <ready_board>,
         @unchanged_numerical_effect <remaining> }

    { @greater_numerical_condition <boarded>,
      @equal_numerical_condition <ready_depart>,
      @greater_numerical_condition <boarded_dest_above> }
    -> { @decrease_numerical_effect <dist_boarded_dest_above>,
         @unchanged_numerical_effect <boarded>,
         @unchanged_numerical_effect <remaining> }

    { @greater_numerical_condition <boarded>,
      @equal_numerical_condition <ready_depart>,
      @equal_numerical_condition <boarded_dest_above>,
      @greater_numerical_condition <boarded_dest_below> }
    -> { @decrease_numerical_effect <dist_boarded_dest_below>,
         @unchanged_numerical_effect <boarded>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <boarded>,
      @equal_numerical_condition <ready_board>,
      @greater_numerical_condition <origin_above> }
    -> { @decrease_numerical_effect <dist_origin_above>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <boarded>,
      @equal_numerical_condition <ready_board>,
      @equal_numerical_condition <origin_above>,
      @greater_numerical_condition <origin_below> }
    -> { @decrease_numerical_effect <dist_origin_below>,
         @unchanged_numerical_effect <remaining> }"""


LOGISTICS_POLICY_DESCRIPTION = """[boolean_features]

[numerical_features]
    <remaining> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_negation
                    @concept_same_as
                        @role_atomic_state "at"
                        @role_atomic_goal "at" true

    <carried> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_existential_quantification
                    @role_atomic_state "in"
                    @concept_union
                        @concept_atomic_state "truck"
                        @concept_atomic_state "airplane"

    <ready_unload> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_union
                            @concept_atomic_state "truck"
                            @concept_atomic_state "airplane"
                    @concept_same_as
                        @role_composition
                            @role_atomic_state "in"
                            @role_atomic_state "at"
                        @role_atomic_goal "at" true

    <pkg_same_city> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "location"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_atomic_state "at"
                                @role_atomic_goal "at" true
                        @concept_same_as
                            @role_composition
                                @role_atomic_state "at"
                                @role_atomic_state "in-city"
                            @role_composition
                                @role_atomic_goal "at" true
                                @role_atomic_state "in-city"

    <pkg_diff_city> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "location"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_atomic_state "at"
                                @role_atomic_goal "at" true
                        @concept_negation
                            @concept_same_as
                                @role_composition
                                    @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"

    <truck_at_pkg_same> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "location"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_atomic_state "at"
                                @role_atomic_goal "at" true
                        @concept_intersection
                            @concept_same_as
                                @role_composition
                                    @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"
                            @concept_existential_quantification
                                @role_atomic_state "at"
                                @concept_existential_quantification
                                    @role_inverse
                                        @role_atomic_state "at"
                                    @concept_atomic_state "truck"

    <truck_at_pkg_diff> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "location"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_composition
                                    @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"
                        @concept_existential_quantification
                            @role_atomic_state "at"
                            @concept_existential_quantification
                                @role_inverse
                                    @role_atomic_state "at"
                                @concept_atomic_state "truck"

    <in_truck_same> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_atomic_state "truck"
                    @concept_same_as
                        @role_composition
                            @role_composition
                                @role_atomic_state "in"
                                @role_atomic_state "at"
                            @role_atomic_state "in-city"
                        @role_composition
                            @role_atomic_goal "at" true
                            @role_atomic_state "in-city"

    <in_truck_diff> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_atomic_state "truck"
                    @concept_negation
                        @concept_same_as
                            @role_composition
                                @role_composition
                                    @role_atomic_state "in"
                                    @role_atomic_state "at"
                                @role_atomic_state "in-city"
                            @role_composition
                                @role_atomic_goal "at" true
                                @role_atomic_state "in-city"
    <truck_airport_diff> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_atomic_state "truck"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_composition
                                    @role_composition
                                        @role_atomic_state "in"
                                        @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"
                        @concept_existential_quantification
                            @role_composition
                                @role_atomic_state "in"
                                @role_atomic_state "at"
                            @concept_atomic_state "airport"

    <pkg_airport_diff> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "airport"
                    @concept_negation
                        @concept_same_as
                            @role_composition
                                @role_atomic_state "at"
                                @role_atomic_state "in-city"
                            @role_composition
                                @role_atomic_goal "at" true
                                @role_atomic_state "in-city"

    <plane_at_pkg_airport> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "airport"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_composition
                                    @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"
                        @concept_existential_quantification
                            @role_atomic_state "at"
                            @concept_existential_quantification
                                @role_inverse
                                    @role_atomic_state "at"
                                @concept_atomic_state "airplane"

    <in_plane_notdest> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_atomic_state "airplane"
                    @concept_negation
                        @concept_same_as
                            @role_composition
                                @role_composition
                                    @role_atomic_state "in"
                                    @role_atomic_state "at"
                                @role_atomic_state "in-city"
                            @role_composition
                                @role_atomic_goal "at" true
                                @role_atomic_state "in-city"

    <in_plane_dest> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "in"
                        @concept_atomic_state "airplane"
                    @concept_same_as
                        @role_composition
                            @role_composition
                                @role_atomic_state "in"
                                @role_atomic_state "at"
                            @role_atomic_state "in-city"
                        @role_composition
                            @role_atomic_goal "at" true
                            @role_atomic_state "in-city"

    <pkg_dest_airport> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "airport"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_atomic_state "at"
                                @role_atomic_goal "at" true
                        @concept_same_as
                            @role_composition
                                @role_atomic_state "at"
                                @role_atomic_state "in-city"
                            @role_composition
                                @role_atomic_goal "at" true
                                @role_atomic_state "in-city"

    <truck_at_dest_airport> ::= 
        @numerical_count
            @concept_intersection
                @concept_intersection
                    @concept_atomic_state "obj"
                    @concept_existential_quantification
                        @role_atomic_goal "at"
                        @concept_atomic_state "location"
                @concept_intersection
                    @concept_existential_quantification
                        @role_atomic_state "at"
                        @concept_atomic_state "airport"
                    @concept_intersection
                        @concept_negation
                            @concept_same_as
                                @role_atomic_state "at"
                                @role_atomic_goal "at" true
                        @concept_intersection
                            @concept_same_as
                                @role_composition
                                    @role_atomic_state "at"
                                    @role_atomic_state "in-city"
                                @role_composition
                                    @role_atomic_goal "at" true
                                    @role_atomic_state "in-city"
                            @concept_existential_quantification
                                @role_atomic_state "at"
                                @concept_existential_quantification
                                    @role_inverse
                                        @role_atomic_state "at"
                                    @concept_atomic_state "truck"
    [policy_rules]
    { @greater_numerical_condition <ready_unload> }
    -> { @decrease_numerical_effect <remaining>,
         @decrease_numerical_effect <ready_unload>,
         @decrease_numerical_effect <carried> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <truck_at_pkg_same>,
      @equal_numerical_condition <truck_at_pkg_diff>,
      @equal_numerical_condition <truck_at_dest_airport>,
      @equal_numerical_condition <pkg_airport_diff>,
      @equal_numerical_condition <pkg_dest_airport>,
      @greater_numerical_condition <pkg_same_city> }
    -> { @increase_numerical_effect <truck_at_pkg_same>,
         @unchanged_numerical_effect <pkg_same_city> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <truck_at_pkg_same>,
      @equal_numerical_condition <truck_at_pkg_diff>,
      @equal_numerical_condition <truck_at_dest_airport>,
      @equal_numerical_condition <pkg_airport_diff>,
      @greater_numerical_condition <pkg_diff_city> }
    -> { @increase_numerical_effect <truck_at_pkg_diff>,
         @unchanged_numerical_effect <pkg_diff_city> }

    { @equal_numerical_condition <carried>,
      @greater_numerical_condition <pkg_airport_diff>,
      @equal_numerical_condition <plane_at_pkg_airport> }
    -> { @increase_numerical_effect <plane_at_pkg_airport>,
         @unchanged_numerical_effect <pkg_airport_diff> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <truck_at_pkg_same>,
      @equal_numerical_condition <truck_at_pkg_diff>,
      @equal_numerical_condition <truck_at_dest_airport>,
      @greater_numerical_condition <pkg_dest_airport> }
    -> { @increase_numerical_effect <truck_at_dest_airport>,
         @unchanged_numerical_effect <pkg_dest_airport> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <pkg_airport_diff>,
      @greater_numerical_condition <pkg_same_city>,
      @greater_numerical_condition <truck_at_pkg_same> }
    -> { @decrease_numerical_effect <pkg_same_city>,
         @increase_numerical_effect <carried>,
         @increase_numerical_effect <in_truck_same>,
         @unchanged_numerical_effect <remaining> }

    { @greater_numerical_condition <in_truck_same>,
      @equal_numerical_condition <ready_unload> }
    -> { @increase_numerical_effect <ready_unload>,
         @unchanged_numerical_effect <carried>,
         @unchanged_numerical_effect <in_truck_same> }

    { @equal_numerical_condition <carried>,
      @equal_numerical_condition <pkg_airport_diff>,
      @greater_numerical_condition <pkg_diff_city>,
      @greater_numerical_condition <truck_at_pkg_diff> }
    -> { @decrease_numerical_effect <pkg_diff_city>,
         @increase_numerical_effect <carried>,
         @increase_numerical_effect <in_truck_diff>,
         @unchanged_numerical_effect <remaining> }

    { @greater_numerical_condition <in_truck_diff>,
      @equal_numerical_condition <truck_airport_diff> }
    -> { @increase_numerical_effect <truck_airport_diff>,
         @unchanged_numerical_effect <carried>,
         @unchanged_numerical_effect <in_truck_diff> }

    { @greater_numerical_condition <truck_airport_diff> }
    -> { @decrease_numerical_effect <carried>,
         @decrease_numerical_effect <in_truck_diff>,
         @increase_numerical_effect <pkg_airport_diff>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <carried>,
      @greater_numerical_condition <pkg_airport_diff>,
      @greater_numerical_condition <plane_at_pkg_airport> }
    -> { @decrease_numerical_effect <pkg_airport_diff>,
         @increase_numerical_effect <carried>,
         @increase_numerical_effect <in_plane_notdest>,
         @unchanged_numerical_effect <remaining> }

    { @greater_numerical_condition <in_plane_notdest> }
    -> { @decrease_numerical_effect <in_plane_notdest>,
         @increase_numerical_effect <in_plane_dest>,
         @unchanged_numerical_effect <carried> }

    { @greater_numerical_condition <in_plane_dest>,
      @equal_numerical_condition <ready_unload> }
    -> { @decrease_numerical_effect <carried>,
         @decrease_numerical_effect <in_plane_dest>,
         @increase_numerical_effect <pkg_dest_airport>,
         @unchanged_numerical_effect <remaining> }

    { @equal_numerical_condition <carried>,
      @greater_numerical_condition <pkg_dest_airport>,
      @greater_numerical_condition <truck_at_dest_airport> }
    -> { @decrease_numerical_effect <pkg_dest_airport>,
         @increase_numerical_effect <carried>,
         @increase_numerical_effect <in_truck_same>,
         @unchanged_numerical_effect <remaining> }"""