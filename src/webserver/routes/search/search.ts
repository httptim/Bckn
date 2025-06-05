/*
 * Copyright 2016 - 2024 Drew Edwards, tmpim
 *
 * This file is part of Krist.
 *
 * Krist is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Krist is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Krist. If not, see <http://www.gnu.org/licenses/>.
 *
 * For more project information, see <https://github.com/tmpim/krist>.
 */

import { Router } from "express";
import { isObject } from "lodash-es";
import { addressToJson, getAddress } from "../../../krist/addresses/index.js";
import { blockToJson, getBlock } from "../../../krist/blocks/index.js";
import { getName, nameToJson } from "../../../krist/names/index.js";
import { getTransaction, transactionToJson } from "../../../krist/transactions/index.js";
import { ReqSearchQuery } from "./index.js";
import { parseQuery, validateQuery } from "./utils.js";

async function performSearch(query: string) {
  const parsed = parseQuery(query);
  const {
    matchAddress, matchName, matchBlock, matchTransaction,
    strippedName, cleanID
  } = parsed;

  const [exactAddress, exactName, exactBlock, exactTransaction] = await Promise.all([
    // exactAddress
    matchAddress ? getAddress(query) : false,
    // exactName
    matchName ? getName(strippedName) : false,
    // exactBlock
    matchBlock && cleanID !== undefined ? getBlock(cleanID) : false,
    // exactBlock
    matchTransaction && cleanID !== undefined ? getTransaction(cleanID) : false
  ]);

  return {
    query: parsed,
    matches: {
      exactAddress: isObject(exactAddress)
        ? addressToJson(exactAddress) : false,
      exactName: isObject(exactName)
        ? nameToJson(exactName) : false,
      exactBlock: isObject(exactBlock)
        ? blockToJson(exactBlock) : false,
      exactTransaction: isObject(exactTransaction)
        ? transactionToJson(exactTransaction) : false
    }
  };
}

export default (): Router => {
  const router = Router();

  /**
   * @api {get} /search Search the Bacon network
   * @apiName Search
   * @apiGroup LookupGroup
   * @apiVersion 2.8.0
   *
   * @apiDescription Search the Bacon network for objects that match the given query, including addresses, names, and
   * transactions.
   *
   * - Addresses are searched by exact address match only
   * - Names are searched by their name with and without the `.bcn` suffix
   * - Transactions are searched by ID
   *
   * For more advanced transaction searches (by involved addresses and metadata), see the `/search/extended` endpoint.
   *
	 * @apiQuery {String} q The search query.
   *
   * @apiUse SearchQuery
   *
   * @apiSuccess {Object} matches The results of the search query.
   * @apiSuccess {Object} matches.exactAddress An exact address match - this will be an Address object if the query
   *   looked like a valid Bacon address, and that address exists in the database. Otherwise, if there is no result, it
   *   will be `false`.
   * @apiSuccess {Object} matches.exactName An exact name match - this will be a Name object if the query looked like a
   *   valid Bacon name (with or without the `.bcn` suffix), and that name exists in the database. Otherwise, if there
   *   is no result, it will be `false`.
   * @apiSuccess {Object} matches.exactBlock Currently unused.
   * @apiSuccess {Object} matches.exactTransaction An exact transaction match - this will be a Transaction object if the
   *   query looked like a valid Bacon transaction ID, and that transaction exists in the database. Otherwise, if there
   *   is no result, it will be `false`.
   *
   * @apiSuccessExample {json} Success - Name result
   * {
   *   "ok": true,
   *   "query": {
   *     "originalQuery": "example",
   *     "matchAddress": false,
   *     "matchName": true,
   *     "matchBlock": false,
   *     "matchTransaction": false,
   *     "strippedName": "example",
   *     "hasID": false
   *   },
   *   "matches": {
   *     "exactAddress": false,
   *     "exactName": {
   *       "name": "example",
   *       "owner": "bxxxxxxxxx",
   *       "registered": "2015-05-24T00:49:04.000Z",
   *       "updated": "2020-01-04T05:09:11.000Z",
   *       "a": null
   *     },
   *     "exactBlock": false,
   *     "exactTransaction": false
   *   }
   * }
   *
   * @apiSuccessExample {json} Success - ID lookup result
   * {
   *   "ok": true,
   *   "query": {
   *     "originalQuery": "1234",
   *     "matchAddress": false,
   *     "matchName": true,
   *     "matchBlock": true,
   *     "matchTransaction": true,
   *     "strippedName": "1234",
   *     "hasID": true,
   *     "cleanID": 1234
   *   },
   *   "matches": {
   *     "exactAddress": false,
   *     "exactName": {
   *       "name": "1234",
   *       "owner": "brazedrugz",
   *       "registered": "2016-10-07T15:55:48.000Z",
   *       "updated": "2016-10-07T15:55:48.000Z",
   *       "a": null
   *     },
   *     "exactBlock": {
   *       "height": 1234,
   *       "address": "2bbb037a6f",
   *       "hash": "01b1b4b7162ec67061760a0f013282b34053b803ad85181d696e8767ed4fa442",
   *       "short_hash": "01b1b4b7162e",
   *       "value": 50,
   *       "time": "2015-02-15T05:37:44.000Z",
   *       "difficulty": 2000000000000
   *     },
   *     "exactTransaction": {
   *       "id": 1234,
   *       "from": null,
   *       "to": "2bbb037a6f",
   *       "value": 50,
   *       "time": "2015-02-15T05:37:40.000Z",
   *       "name": null,
   *       "metadata": null,
   *       "sent_metaname": null,
   *       "sent_name": null,
   *       "type": "mined"
   *     }
   *   }
   * }
   */
  router.get("/", async (req: ReqSearchQuery, res) => {
    const query = validateQuery(req);
    const results = await performSearch(query);
    res.json({
      ok: true,
      ...results
    });
  });

  return router;
};
